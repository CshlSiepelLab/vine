/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <phast/misc.h>
#include <phast/trees.h>
#include <nj.h>
#include <likelihoods.h>
#include <gradients.h>
#include <mcmc.h>
#include <multi_mvn.h>
#include <variational.h>
#include <geometry.h>

/* MCMC refinement of variational samples */

List *var_sample_mcmc(int nsamples, int thin, multi_MVN *mmvn,
                         CovarData *data, TreeModel *mod, FILE *logf,
                         unsigned int silent) {

  if (thin <= 0)
    die("ERROR in var_sample_mcmc: --thin must be >= 1 (got %d).\n",
        thin);

  int n = data->nseqs, dim = data->dim, fulld = n * dim;
  int niters = 0, naccept = 0, last_naccept = 0, nsamp = 0;
  unsigned int keep_sampling = TRUE, burnin = TRUE, accept = FALSE;
  unsigned int first_iter = TRUE;  /* MH always accepts the first proposal */
  /* rho chosen so thinned samples have ~50% autocorrelation, giving
   * ESS >= n/3.  Target: (1 - alpha*(1-rho))^thin = 0.5, solved for rho.
   * For small thin this may be 0 (independence sampler). */
  double rho = fmax(0.0, 1.0 - log(2.0) / (thin * TARGET_ACCEPT_RATE));

  /* optimal RW Metropolis scaling: effective z-step = sqrt(1-rho²)*s should
   * equal 2.38/sqrt(d); solve for s: s_init = 2.38/(sqrt(1-rho²)*sqrt(d)) */
  double s_init = (rho < 1.0)
    ? fmin(10.0, 2.38 / (sqrt(1.0 - rho * rho) * sqrt((double)fulld)))
    : 1.0;
  double s = s_init, accrt = 0.0, new_accrt = 0.0;
  Vector *mu = vec_new(fulld);
  double nf_logdet;
  mmvn_save_mu(mmvn, mu);
  TreeNode *tree = NULL, *oldtree = NULL;
  List *retval = lst_new_ptr(nsamples);

  double lnl = 0.0, lastlnl = 0.0, avelnl = 0.0;
  double *lnl_samps = (double *)smalloc(nsamples * sizeof(double));

  if (logf != NULL) {
    fprintf(logf, "### Starting MCMC sampling at posterior mean (burn-in of %d iterations, target acceptance rate %.3f, rho=%.4f, s_init=%.4f)\n",
      BURNIN_ITERS, TARGET_ACCEPT_RATE, rho, s_init);
    fprintf(logf, "## %s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "iter", "prop_lnl",
            "acc_lnl", "burnin", "accept", "retain", "block_accrt", "tot_accrt",
            "s");
  }

  /* start with mod->tree == NULL and free any previous tree; all
   * persistent tree handling will be with tree and oldtree */
  if (mod->tree != NULL) {
    tr_free(mod->tree);
    mod->tree = NULL;
  }
  
  Vector *lastz = vec_new(fulld), *zprop = vec_new(fulld),
    *zeta = vec_new(fulld), *x = vec_new(fulld), *y = vec_new(fulld);
  vec_zero(lastz);

  while (keep_sampling) {
    niters++;

    if (niters == BURNIN_ITERS) {
      burnin = FALSE;
      if (!silent) fprintf(stderr, "Burn-in of %d iterations complete...\n", BURNIN_ITERS);
    }      
    
    /* propose new z by preconditioned Crank-Nicolson */
    vec_copy(zprop, lastz);
    vec_scale(zprop, rho);
    mvn_sample_std(zeta);
    vec_scale(zeta, sqrt(1 - rho * rho));
    vec_plus_eq(zprop, zeta);

    /* propose new x centered on variational mean using variational covariance;
     * param s scales overall step size and is tuned dynamically (below).
     * For non-LOWR types: x = mu + s*L*z (mmvn_map_std applies L and adds mu).
     * For LOWR: fall back to isotropic x = mu + s*z */
    vec_copy(x, zprop);
    vec_scale(x, s);
    if (mmvn->type != MVN_LOWR)
      mmvn_map_std(mmvn, x);
    else
      vec_plus_eq(x, mu);
    
    /* convert x to y using normalizing flows if available */
    apply_normalizing_flows(y, x, data, &nf_logdet);

    /* convert to tree */
    points_to_distances(y, data);
    tree = infer_distance_tree(data->dist, data->names, NULL, NULL, data);
    mod->tree = NULL; /* prevent reset_tree_model from freeing the tree */
    reset_tree_model(mod, tree);

    /* calculate log likelihood */
    if (data->crispr_mod != NULL)
      lnl = cpr_compute_log_likelihood(data->crispr_mod, NULL);
    else
      lnl = compute_log_likelihood(mod, data, NULL);
    
    /* also get migration log likelihood if needed */
    if (data->crispr_mod != NULL && data->migtable != NULL) {
      lnl += mig_compute_log_likelihood(mod, data->migtable, data->crispr_mod,
        NULL);
    }
    
    double alpha = first_iter ? 1.0 : fmin(1.0, exp(lnl - lastlnl));
    if (unif_rand() < alpha) {
      accept = TRUE;
      first_iter = FALSE;
      lastlnl = lnl;
      vec_copy(lastz, zprop);
      naccept++;
      if (oldtree != NULL) {
        tr_free(oldtree); /* free old tree */
        oldtree = NULL;
      }
    }
    else {
      accept = FALSE;
      tr_free(tree); /* free rejected tree */
      tree = oldtree; /* point to old tree */
      oldtree = NULL; 
    }

    /* now 'tree' contains the current state of the chain, whether we
     * accepted or not; secondary tree has been freed and oldtree == NULL */

    accrt = naccept / (double)niters;

    /* keep track of block-wise acceptance rate */
    if (niters % TUNING_INTERVAL == 0) {
      int new_naccept =
          naccept - last_naccept; /* number accepted since last check */
      int t = niters / TUNING_INTERVAL; /* number of checks so far */
      new_accrt =
        new_naccept / (double)TUNING_INTERVAL; /* acceptance rate since last check */

      /* during burnin only, adapt s to target acceptance rate;
       * rho is fixed (see s_init comment above); after burnin, keep fixed */
      if (burnin) {
        /* Diminishing rate: 0.5/sqrt(t) reaches the optimal s in ~5 blocks
         * (vs ~37 for 0.2/sqrt(t)).  Worst-case drift with 100% acceptance
         * over 100 blocks: exp(0.5 * 2*sqrt(100) * 0.7) = exp(7) ≈ 1100x,
         * but the s<=10 cap and self-correcting acceptance keep it bounded. */
        double eta_s = 0.5 / sqrt((double)t);
        s = exp(log(s) + eta_s * (new_accrt - TARGET_ACCEPT_RATE));
        if (s < MIN_S) s = MIN_S;
        if (s > 10)
          s = 10;
      }

      last_naccept = naccept;
    }

    if (!burnin && niters % thin == 0) {      
      if (tree == NULL) {
        /* special case where all iterations since last collect were
         * rejections; reconstruct current state tree from lastz for
         * output */
        vec_copy(x, lastz);
        vec_scale(x, s);
        if (mmvn->type != MVN_LOWR)
          mmvn_map_std(mmvn, x);
        else
          vec_plus_eq(x, mu);
        apply_normalizing_flows(y, x, data, &nf_logdet);
        points_to_distances(y, data);
        tree = infer_distance_tree(data->dist, data->names, NULL, NULL, data);
        mod->tree = NULL; /* clear dangling ptr left by rejected proposed tree */
      }
      
      lnl_samps[nsamp] = lastlnl;
      lst_push_ptr(retval, tree);
      tree = NULL; /* prevent freeing below */
      nsamp++;
      avelnl += lastlnl;
      
      if (nsamp >= nsamples)
        keep_sampling = FALSE;

      /* progress line every fifth of the run; clamp the divisor to >=1
         so --nsamples<5 does not modulo-by-zero (SIGFPE) */
      int prog_every = nsamples / 5;
      if (prog_every < 1) prog_every = 1;
      if (nsamp % prog_every == 0 && !silent)
        fprintf(stderr, "Collected %d/%d samples...\n", nsamp, nsamples);
    }

    if (logf != NULL)
      fprintf(logf, "## %d\t%f\t%f\t%u\t%u\t%u\t%f\t%f\t%f\n", niters, lnl,
              lastlnl, burnin, accept, (!burnin && niters % thin == 0),
              new_accrt, accrt, s);    

    oldtree = tree; /* set up for next iteration */
  }

  /* note that the last tree sampled must have been retained in
   * retval, so we don't have to worry about freeing it here; all
   * other trees have been freed during the loop. */

  /* compute ESS from log-likelihood autocorrelation of collected samples
   * (Geyer initial positive sequence: truncate at first non-positive lag) */
  double ess = (double)nsamp;
  if (nsamp > 1) {
    int i, k;
    double mean = 0, var = 0, sum_rho = 0;
    for (i = 0; i < nsamp; i++) mean += lnl_samps[i];
    mean /= nsamp;
    for (i = 0; i < nsamp; i++) var += (lnl_samps[i] - mean) * (lnl_samps[i] - mean);
    var /= nsamp;
    if (var > 0) {
      for (k = 1; k < nsamp; k++) {
        double rho_k = 0;
        for (i = 0; i < nsamp - k; i++)
          rho_k += (lnl_samps[i] - mean) * (lnl_samps[i+k] - mean);
        rho_k /= ((nsamp - k) * var);
        if (rho_k <= 0) break;
        sum_rho += rho_k;
      }
      ess = nsamp / (1.0 + 2.0 * sum_rho);
    }
  }
  avelnl /= nsamples;

  if (!silent)
    fprintf(stderr,
            "MCMC sampling complete; acceptance rate %.3f over %d iterations, "
            "ESS %.1f/%d (%.0f%%), average lnl %.2f...\n",
            accrt, niters, ess, nsamp, 100.0 * ess / nsamp, avelnl);
  
  if (logf != NULL)
    fprintf(logf,
            "### Final MCMC stats: %d iterations, acceptance rate %.2f, "
            "ESS %.1f/%d (%.0f%%), avelnl %.3f\n",
            niters, accrt, ess, nsamp, 100.0 * ess / nsamp, avelnl);
  
  sfree(lnl_samps);

  mod->tree = NULL; /* avoid dangling pointer */
  vec_free(mu);
  vec_free(lastz);
  vec_free(zprop);
  vec_free(zeta);
  vec_free(x);
  vec_free(y);
  
  return retval; 
}
