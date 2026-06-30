/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* core variational inference routines */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <variational.h>
#include <nj.h>
#include <upgma.h>
#include <geometry.h>
#include <adam_scheduler.h>
#include <sparse_matrix.h>
#include <gradients.h>
#include <nuisance.h>
#include <likelihoods.h>
#include <hutchinson.h>
#include <version.h>

/* number of warmup iterations to run with migration model disabled;
   allows tree topology to converge before migration inference activates */
#define CPR_MIG_WARMUP_ITERS 150

static double nj_rescale_mean_grad_el(Vector *grad, multi_MVN *mmvn,
                                      CovarData *data, int i) {
  if (data->natural_grad != TRUE)
    return vec_get(grad, i);

  if (data->type == CONST || data->type == DIAG)
    return vec_get(grad, i) * mat_get(mmvn->mvn->sigma, i, i);
  else {
    double dotp = 0.0;
    int sigmarow = i / mmvn->d, d = i % mmvn->d;
    for (int j = 0; j < mmvn->mvn->sigma->ncols; j++)
      dotp += mat_get(mmvn->mvn->sigma, sigmarow, j) *
        vec_get(grad, j*mmvn->d + d);
    return dotp;
  }
}

static double nj_elbo_mix_kld_montecarlo(mixture_MVN *mixmvn, int component,
                                         CovarData *data, int nminibatch,
                                         double *kld,
                                         Vector *sigma_kldgrad,
                                         Vector **mu_kldgrad);

/* optimize variational model by stochastic gradient ascent using the
   Adam algorithm.  Takes initial tree model and alignment and
   distance matrix, dimensionality of Euclidean space to work in.
   Note: alters distance matrix */
void nj_variational_inf(TreeModel *mod, mixture_MVN *mixmvn, int nminibatch,
                        double learnrate, int nbatches_conv, int min_nbatches,
                        CovarData *data, FILE *logf,
                        unsigned int silent, unsigned int log_all) {

  Vector *model_grad;          /* selected-component mu + shared sigma likelihood/prior gradient */
  Vector *model_natgrad;       /* natural-gradient-scaled model_grad after sigma regularizers */
  Vector *sigma_kldgrad;       /* shared covariance KLD gradient */
  Vector *sigma_penalty_grad;  /* variance-penalty gradient, sigma block only */
  Vector **mu_kldgrad = NULL;  /* per-component KLD mean gradients */
  Vector *m_sigma, *m_sigma_prev, *v_sigma, *v_sigma_prev,
    *best_sigmapar, *sigmapar = data->params;
  Vector **m_mu = NULL, **v_mu = NULL, **best_mu = NULL;
  int n = data->nseqs, j, k, t, stop = FALSE, bestt = -1, graddim,
    dim = data->dim, fulld = n*dim, reenable_taylor_t = -1,
    component = 0, ncomponents = mixmvn->ncomponents;
  int *component_t = NULL;
  double elb = 0, avell, avemigll, kld, bestelb = -INFTY, bestll = -INFTY,
    bestkld = -INFTY, bestmigll = -INFTY,
    running_tot = 0, last_running_tot = -INFTY, trace, logdet, penalty = 0,
    bestpenalty = 0, ave_lprior, best_lprior = -INFTY, subsamp_rescale = 1.0,
    ll_at_mean = 0, bestll_at_mean = -INFTY, grad_norm_sq;
  multi_MVN *mmvn = mixmvn_get_component(mixmvn, component);  /* Initialize mmvn with the first component */
  TaylorData *taylor_stash = NULL;

  /* for nuisance parameters; these are parameters that are optimized
     by stochastic gradient descent but are not fully sampled via the
     variational distribution */
  int n_nuisance_params = nj_get_num_nuisance_params(mod, data);
  Vector *ave_nuis_grad = NULL, *m_nuis = NULL, *v_nuis = NULL,
    *m_nuis_prev = NULL, *v_nuis_prev = NULL, *best_nuis_params = NULL,
    *center = NULL;
  if (mmvn->d * mmvn->n != dim * n)
    die("ERROR in nj_variational_inf: bad dimensions\n");

  graddim = fulld + data->params->size;
  sigma_kldgrad = vec_new(sigmapar->size);
  model_grad = vec_new(graddim);
  model_natgrad = vec_new(graddim);
  m_sigma = vec_new(sigmapar->size);
  m_sigma_prev = vec_new(sigmapar->size);
  v_sigma = vec_new(sigmapar->size);
  v_sigma_prev = vec_new(sigmapar->size);
  sigma_penalty_grad = vec_new(graddim); /* full-shaped for helper API;
                                            only sigma block is populated */
  component_t = smalloc(ncomponents * sizeof(int));
  m_mu = smalloc(ncomponents * sizeof(Vector*));
  v_mu = smalloc(ncomponents * sizeof(Vector*));
  best_mu = smalloc(ncomponents * sizeof(Vector*));
  mu_kldgrad = smalloc(ncomponents * sizeof(Vector*));

  if (n_nuisance_params > 0) {
    ave_nuis_grad = vec_new(n_nuisance_params);
    m_nuis = vec_new(n_nuisance_params);
    v_nuis = vec_new(n_nuisance_params);
    m_nuis_prev = vec_new(n_nuisance_params);
    v_nuis_prev = vec_new(n_nuisance_params);
    best_nuis_params = vec_new(n_nuisance_params);
  }
  
  best_sigmapar = vec_new(sigmapar->size);
  vec_copy(best_sigmapar, sigmapar);
  center = vec_new(dim);

  /* initialize component-specific moments and iteration counts*/
  for (k = 0; k < ncomponents; k++) {
    m_mu[k] = vec_new(fulld);
    v_mu[k] = vec_new(fulld);
    best_mu[k] = vec_new(fulld);
    mu_kldgrad[k] = vec_new(fulld);
    vec_zero(m_mu[k]);
    vec_zero(v_mu[k]);
    vec_zero(mu_kldgrad[k]);
    mmvn_save_mu(mixmvn_get_component(mixmvn, k), best_mu[k]);
    component_t[k] = 0;
  }

  /* set up log file */
  if (logf != NULL) {
    fprintf(logf, "state\tll\telbo\t");
    if (data->treeprior != NULL)
      fprintf(logf, "prior\t");
    else
      fprintf(logf, "kld\t");
    if (data->taylor)
      fprintf(logf, "half_trHS\telbo_bias\t");
    if (data->var_reg != 0)
      fprintf(logf, "penalty\t");
    if (data->crispr_mod == NULL)
      fprintf(logf, "subsamp\treuse\tgradnorm\tclip\t");
    if (data->migtable != NULL)
      fprintf(logf, "mig_ll\t");
    if (log_all) {
      for (j = 0; j < fulld; j++)
        fprintf(logf, "mu.%d\t", j);
      if (data->type == LOWR || data->type == DIAG) {
        for (j = 0; j < sigmapar->size; j++)
          fprintf(logf, "sigma.%d\t", j);
      }
    }
    if (data->type == CONST || data->type == DIST) {
      for (j = 0; j < sigmapar->size; j++)
        fprintf(logf, "sigma.%d\t", j);
    }
    for (j = 0; j < n_nuisance_params; j++)
      fprintf(logf, "%s\t", nj_get_nuisance_param_name(mod, data, j));
    fprintf(logf, "\n");
  }

  /* initialize moments for Adam algorithm */
  vec_zero(m_sigma);  vec_zero(m_sigma_prev);
  vec_zero(v_sigma);  vec_zero(v_sigma_prev);
  if (n_nuisance_params > 0) {
    vec_zero(m_nuis);  vec_zero(m_nuis_prev);
    vec_zero(v_nuis);  vec_zero(v_nuis_prev);
  }
  t = 0;

  /* set up scheduler; for CRISPR mode, start in full mode (no
     subsampling) but still use adaptive gradient clipping */
  int maxlen = data->crispr_mod == NULL ?
    data->msa->length : data->crispr_mod->nsites;
  int init_subsamp = data->crispr_mod == NULL ? NSUBSAMPLES : maxlen;
  Scheduler *s = sched_new(maxlen, init_subsamp, 20,
                           learnrate, 10, 50, 30);
  SchedState *st = sched_new_state(s);
  SchedDirectives *sd = smalloc(sizeof(SchedDirectives));
  SchedMetrics *sm = smalloc(sizeof(SchedMetrics));
  sm->grad_norm = 0;

  do {

    /* Sample which component to use this iteration */
    component = mixmvn_sample_component(mixmvn);
    mmvn = mixmvn_get_component(mixmvn, component);

    /* simple update to user */
    if (t > 0 && t % 100 == 0) {
      if (!silent) {
        fprintf(stderr, "Iteration %d", t);
        if (bestelb > -INFTY)
          fprintf(stderr, "; best ELBO=%.2f", bestelb);
        fprintf(stderr, "...\n");
      }
    }
    
    /* get directives from scheduler */
    sched_next(s, st, sm, sd);
    unsigned int clipped = FALSE;
    double clip_scale = 1.0;
    
    /* we can precompute the KLD because it does not depend on the data under this model,
       (see equation 7, Doersch arXiv 2016). This does not apply to the mixture case, which
       will be handled using monte carlo below. */
    kld = 0;
    vec_zero(sigma_kldgrad);
    for (k = 0; k < ncomponents; k++)
      vec_zero(mu_kldgrad[k]);
    if (ncomponents == 1) {
      logdet = mmvn_log_det(mmvn);
      if (data->treeprior == NULL) { /* only do if no explicit tree prior */
        trace = mmvn_trace(mmvn);  /* we'll reuse this */
      
        kld = 0.5 * (trace + mmvn_mu2(mmvn) - fulld - logdet);

        kld *= data->kld_upweight/(data->pointscale*data->pointscale);      
      
        /* we can also precompute the contribution of the KLD to the gradient */
        /* Note KLD is subtracted rather than added, so compute the gradient of -KLD */
        for (j = 0; j < fulld; j++)
          vec_set(mu_kldgrad[0], j, -1.0 * mmvn_get_mu_el(mmvn, j));
        for (j = 0; j < sigmapar->size; j++) {
          double gj = 0.0;

          /* partial deriv wrt sigma_j is more complicated because of
             the trace and log determinant */
          if (data->type == CONST || data->type == DIST)
            gj = 0.5 * (fulld - trace);
          else if (data->type == DIAG) 
            gj = 0.5 * (1.0 - mat_get(mmvn->mvn->sigma, j, j)); 
          else 
            continue; /* LOWR case is messy; handle below */
          vec_set(sigma_kldgrad, j, gj);
        }
      
        if (data->type == LOWR) 
          nj_set_kld_sigma_grad_LOWR(sigma_kldgrad, mmvn);
      }
      else { /* with explicit tree prior, we need the entropy of the MVN instead */
        kld = -0.5 * (fulld * (1.0 + log(2 * M_PI)) + logdet);
        kld *= data->kld_upweight/(data->pointscale*data->pointscale);      
        /* note overloading name and negating */
        for (j = 0; j < sigmapar->size; j++) {
          double gj = 0.0;

          /* partial deriv wrt mu_j is zero; only sigma contributes. */
          if (data->type == CONST || data->type == DIST)
            gj = 0.5 * fulld;
          else if (data->type == DIAG) 
            gj = 0.5;
          else 
            continue; /* LOWR case is messy; handle below */
          vec_set(sigma_kldgrad, j, gj);
        }
        if (data->type == LOWR) 
          nj_set_entropy_sigma_grad_LOWR(sigma_kldgrad, mmvn);
      }
    }

    /* can also pre-compute variance penalty, which is okay in the mixture case
    when the covariance is shared across components */
    vec_zero(sigma_penalty_grad);
    nj_compute_variance_penalty(sigma_penalty_grad, mmvn, data);
    penalty = data->var_pen;

    if (ncomponents == 1) {
      vec_scale(sigma_kldgrad,
                data->kld_upweight/(data->pointscale*data->pointscale));
      vec_scale(mu_kldgrad[0],
                data->kld_upweight/(data->pointscale*data->pointscale));
    }


    /* now estimate ELBO and gradient, either by Monte Carlo integration or by
     * the Taylor approximation */

    /* first set up subsampling based on scheduler parameters (but not
       in crispr mode or with multithreading) */
    if (!sd->full_grad_now && data->crispr_mod == NULL && data->nthreads == 1) {
      data->subsample = TRUE;
      data->subsampsize = sd->m;
      data->reuse_subsamp = !sd->resample_sites;
      subsamp_rescale = (double)data->msa->length / data->subsampsize;
    }
    else { /* no subsampling */
      data->subsample = FALSE;
      subsamp_rescale = 1.0;
    }

    /* check whether to re-enable Taylor approximation */
    if (taylor_stash != NULL && data->taylor == NULL && t == reenable_taylor_t) {
      if (!silent) fprintf(stderr, "WARNING: re-enabling Taylor approximation.\n");
      data->taylor = taylor_stash;
      taylor_stash = NULL;
    }
    
    /* migration warmup: disable migration for first CPR_MIG_WARMUP_ITERS
       iterations to let tree topology converge before migration activates */
    if (data->crispr_mod != NULL && data->migtable != NULL) {
      if (t < CPR_MIG_WARMUP_ITERS) {
        if (t == 0 && !silent)
          fprintf(stderr, "Running %d warmup iterations without migration "
                  "model...\n", CPR_MIG_WARMUP_ITERS);
        data->crispr_mod->mig_warmup = TRUE;
      } else {
        if (t == CPR_MIG_WARMUP_ITERS && !silent)
          fprintf(stderr, "Warmup complete; enabling migration model...\n");
        data->crispr_mod->mig_warmup = FALSE;
      }
    }

    if (data->taylor != NULL) {
      avell = nj_elbo_hybrid(mod, mixmvn, component, data, nminibatch,
                             model_grad, ave_nuis_grad, &ave_lprior, &avemigll,
                             &ll_at_mean);
      /* avell = nj_elbo_taylor(mod, mmvn, data, model_grad, ave_nuis_grad, */
      /*                        &ave_lprior, &avemigll, &ll_at_mean); */
      if ((data->crispr_mod != NULL && data->crispr_mod->zero_likl == TRUE) ||
          !isfinite(avell)) {
        if (!silent) fprintf(stderr, "WARNING: Taylor approximation produced invalid likelihood; "
                "switching to Monte Carlo.\n");
        reenable_taylor_t = t + 10;
        taylor_stash = data->taylor;
        data->taylor = NULL;
      }
      else if (ncomponents > 1) {
        /* Taylor approximates only the likelihood/prior expectation.  The
           mixture entropy term E[log q_mix(x)] has no simple closed form, so
           keep estimating just the mixture KLD and its gradients by MC. */
        nj_elbo_mix_kld_montecarlo(mixmvn, component, data, nminibatch,
                                   &kld, sigma_kldgrad, mu_kldgrad);
      }
    }
    
    if (data->taylor == NULL) {
      ll_at_mean = 0;  /* not available in MC path */
      avell = nj_elbo_montecarlo(mod, mixmvn, component, data, nminibatch,
                                 model_grad, ave_nuis_grad, &ave_lprior,
                                 &avemigll, &kld, sigma_kldgrad, mu_kldgrad);
    }
    
    /* In subsample mode the likelihood (and its gradient) are computed
       on a subsampled set of sites and need rescaling to the full-data
       scale before being combined with the full-scale KLD, sparsity,
       and tree-prior contributions.  Rescaling model_grad/ave_nuis_grad
       as a whole over-amplifies the (much smaller) tree-prior and
       flow-logdet contributions by the same factor; that error is
       bounded and small in magnitude compared to the previous bug
       (LL grad too small relative to regularizers, causing
       over-regularization). */
    if (data->subsample == TRUE) {
      vec_scale(model_grad, subsamp_rescale);
      if (ave_nuis_grad != NULL)
        vec_scale(ave_nuis_grad, subsamp_rescale);
      avell *= subsamp_rescale;
    }

    for (j = 0; j < sigmapar->size; j++)
      vec_set(model_grad, fulld + j,
              vec_get(model_grad, fulld + j) + vec_get(sigma_kldgrad, j));
    vec_plus_eq(model_grad, sigma_penalty_grad);

    /* store parameters if best yet */
    elb = avell + ave_lprior - kld - penalty + avemigll;

    /* don't select best during migration warmup: migration is excluded from
     * the ELBO then, making warmup ELBOs artificially high and incomparable
     * to post-warmup ELBOs that include the migration log likelihood */
    int mig_warmup_active = (data->crispr_mod != NULL && data->migtable != NULL
                             && data->crispr_mod->mig_warmup);
    if (elb > bestelb && (sd->full_grad_now || data->crispr_mod != NULL)
        && !mig_warmup_active) {
      bestelb = elb;
      bestll = avell;  /* not necessarily best ll but ll corresponding to bestelb */
      bestll_at_mean = ll_at_mean;  /* ll at posterior mean; 0 if MC path */
      best_lprior = ave_lprior;
      bestkld = kld;  
      bestpenalty = penalty;
      bestmigll = avemigll;
      bestt = t;
      for (k = 0; k < ncomponents; k++)
        mmvn_save_mu(mixmvn_get_component(mixmvn, k), best_mu[k]);
      vec_copy(best_sigmapar, sigmapar);
      if (n_nuisance_params > 0)
        nj_save_nuis_params(best_nuis_params, mod, data);
    }

    /* rescale gradient by approximate inverse Fisher information to
       put on similar scales; seems to help with optimization */
    if (data->natural_grad == TRUE)
      nj_rescale_grad(model_grad, model_natgrad, mmvn, data);
    else
      vec_copy(model_natgrad, model_grad);
    /* we won't do this with nuisance params */

    /* update scheduler with norm of the actual Adam update and clip if
       necessary.  Mean KLD gradients are stored by component, so include
       them explicitly rather than measuring only model_natgrad. */
    grad_norm_sq = 0.0;
    for (k = 0; k < ncomponents; k++) {
      multi_MVN *kmmvn = mixmvn_get_component(mixmvn, k);
      for (j = 0; j < fulld; j++) {
        double g = k == component ? vec_get(model_natgrad, j) : 0.0;
        g += nj_rescale_mean_grad_el(mu_kldgrad[k], kmmvn, data, j);
        grad_norm_sq += g * g;
      }
    }
    for (j = 0; j < sigmapar->size; j++)
      grad_norm_sq += pow(vec_get(model_natgrad, fulld + j), 2);
    sm->grad_norm = sqrt(grad_norm_sq);
    if (sd->clip_norm > 0 && sm->grad_norm > sd->clip_norm) {
      clip_scale = sd->clip_norm / sm->grad_norm;
      vec_scale(model_natgrad, clip_scale);
      clipped = TRUE;
    }

    /* Adam updates; see Kingma & Ba, arxiv 2014 */
    t++;
    for (k = 0; k < ncomponents; k++)
      component_t[k]++; /* track Adam update count for each component */
    data->variational_iter = t; /* useful for debugging in other routines */

    /* Update mu.  The selected component receives the stochastic
       likelihood/prior gradient; all components receive mixture-KLD mean
       gradients because log q_mix depends on every component density. */
    for (k = 0; k < ncomponents; k++) {
      multi_MVN *kmmvn = mixmvn_get_component(mixmvn, k);
      for (j = 0; j < fulld; j++) {
        double mhatj, vhatj, g = k == component ? vec_get(model_natgrad, j) : 0.0;
        g += clip_scale * nj_rescale_mean_grad_el(mu_kldgrad[k],
                                                  kmmvn, data, j);
        vec_set(m_mu[k], j, ADAM_BETA1 * vec_get(m_mu[k], j) + (1.0 - ADAM_BETA1) * g);
        vec_set(v_mu[k], j, ADAM_BETA2 * vec_get(v_mu[k], j) + (1.0 - ADAM_BETA2) * pow(g, 2));
        mhatj = vec_get(m_mu[k], j) / (1.0 - pow(ADAM_BETA1, component_t[k]));
        vhatj = vec_get(v_mu[k], j) / (1.0 - pow(ADAM_BETA2, component_t[k]));
        mmvn_set_mu_el(kmmvn, j, mmvn_get_mu_el(kmmvn, j) + sd->lr * mhatj / (sqrt(vhatj) + ADAM_EPS));
      }
    }

    /* update sigma (shared across components) */
    for (j = 0; j < sigmapar->size; j++) {
      double mhatj, vhatj, g = vec_get(model_natgrad, fulld + j);
      vec_set(m_sigma, j, ADAM_BETA1 * vec_get(m_sigma_prev, j) + (1.0 - ADAM_BETA1) * g);
      vec_set(v_sigma, j, ADAM_BETA2 * vec_get(v_sigma_prev, j) + (1.0 - ADAM_BETA2) * pow(g, 2));
      mhatj = vec_get(m_sigma, j) / (1.0 - pow(ADAM_BETA1, t));
      vhatj = vec_get(v_sigma, j) / (1.0 - pow(ADAM_BETA2, t));
      vec_set(sigmapar, j, vec_get(sigmapar, j) + sd->lr * mhatj / (sqrt(vhatj) + ADAM_EPS));
    }
    mixmvn_update_covariance(mixmvn, data);
    
    vec_copy(m_sigma_prev, m_sigma);
    vec_copy(v_sigma_prev, v_sigma);

    /* same thing for nuisance params, if necessary */
    for (j = 0; j < n_nuisance_params; j++) {   
      double mhatj_nuis, vhatj_nuis, g = vec_get(ave_nuis_grad, j);
      vec_set(m_nuis, j, ADAM_BETA1 * vec_get(m_nuis_prev, j) + (1.0 - ADAM_BETA1) * g);
      vec_set(v_nuis, j, ADAM_BETA2 * vec_get(v_nuis_prev, j) + (1.0 - ADAM_BETA2) * pow(g,2));
      mhatj_nuis = vec_get(m_nuis, j) / (1.0 - pow(ADAM_BETA1, t));
      vhatj_nuis = vec_get(v_nuis, j) / (1.0 - pow(ADAM_BETA2, t));
      nj_nuis_param_pluseq(mod, data, j, sd->lr * 0.3 * mhatj_nuis / (sqrt(vhatj_nuis) + ADAM_EPS));
      /* factor of 0.3 above to slow learning of nuisance params */
    }
    if (n_nuisance_params > 0) {
      vec_copy(m_nuis_prev, m_nuis);
      vec_copy(v_nuis_prev, v_nuis);
    }
    
    /* report to log file */
    if (logf != NULL) {
      fprintf(logf, "%d\t%f\t%f\t", t, avell, elb);
      if (data->treeprior != NULL)
        fprintf(logf, "%f\t", ave_lprior);
      else
        fprintf(logf, "%f\t", kld);
      if (data->taylor)
        fprintf(logf, "%f\t%f\t", 0.5 * data->taylor->T_cache,
                data->taylor->elbo_bias);
      else if (taylor_stash != NULL)
        fprintf(logf, "0\t0\t"); /* place holder */
      if (data->var_reg != 0)
        fprintf(logf, "%f\t", data->var_pen);
      if (data->crispr_mod == NULL)
        fprintf(logf, "%d\t%d\t%f\t%d\t", data->subsampsize,
                data->reuse_subsamp, sm->grad_norm, clipped);
      if (data->migtable != NULL) 
        fprintf(logf, "%f\t", avemigll); 
      if (log_all) {
        mmvn_print(mmvn, logf, TRUE, FALSE);
        if (data->type == LOWR || data->type == DIAG) {
          for (j = 0; j < sigmapar->size; j++)
            fprintf(logf, "%f\t", vec_get(sigmapar, j));
        }
      }
      if (data->type == CONST || data->type == DIST) {
        for (j = 0; j < sigmapar->size; j++)
          fprintf(logf, "%f\t", vec_get(sigmapar, j));
      }
      for (j = 0; j < n_nuisance_params; j++)
        fprintf(logf, "%f\t", nj_nuis_param_get(mod, data, j));

      fprintf(logf, "\n");
    }
    
    /* check total elb every nbatches_conv to decide whether to stop */
    running_tot += elb;
    if (t % nbatches_conv == 0) {
      if (logf != NULL)
        fprintf(logf, "# Average ELBO for last %d: %f\n", nbatches_conv, running_tot/nbatches_conv);
      if ((sd->full_grad_now || data->crispr_mod != NULL) && t >= min_nbatches &&
          1.001*running_tot <= last_running_tot*0.999)
        /* sometimes get stuck increasingly asymptotically; stop if increase not more than about 0.1% */
        stop = TRUE;

      last_running_tot = running_tot;
      running_tot = 0;
    }    
  } while(stop == FALSE);

  /* Revert to the best parameters found after convergence */
  for (k = 0; k < ncomponents; k++)
    mmvn_set_mu(mixmvn_get_component(mixmvn, k), best_mu[k]);
  vec_copy(sigmapar, best_sigmapar);
  mixmvn_update_covariance(mixmvn, data);
  if (n_nuisance_params > 0)
    nj_update_nuis_params(best_nuis_params, mod, data);

  /* if using Taylor approximation, run one final MC pass at the restored
     best parameters to get an unbiased estimate of E[lnL] for reporting.
     The hybrid ELBO used during training can be biased (especially when
     variance is at floor), so this gives an accurate final value. */
  double final_mc_ll = 0;
  if (data->taylor != NULL && logf != NULL) {
    double dummy_lprior = 0, dummy_migll = 0;
    final_mc_ll = nj_elbo_montecarlo(mod, mixmvn, component, data, nminibatch,
                                     model_grad, ave_nuis_grad, &dummy_lprior,
                                     &dummy_migll, NULL, NULL, NULL);
  }

  if (logf != NULL) {
    fprintf(logf,
            "# Reverting to parameters from iteration %d; ELB: %.2f, LNL: "
            "%.2f, LPRIOR: %.2f, KLD: %.2f, penalty: %.2f",
            bestt + 1, bestelb, bestll, best_lprior, bestkld, bestpenalty);
    if (data->taylor != NULL)
      /* final unbiased MC estimate of E_q[lnL] at best parameters */
      fprintf(logf, ", LNL_mc: %.2f", final_mc_ll);
    else if (bestll_at_mean != 0)
      /* for MC mode: lnL at the mean embedding (no separate MC pass needed) */
      fprintf(logf, ", LNL_mu: %.2f", bestll_at_mean);
    if (data->migtable != NULL)
      fprintf(logf, ", MIGLL: %.2f", bestmigll);
    for (j = 0; j < n_nuisance_params; j++) /* print these also if available */
      fprintf(logf, ", %s: %.4f", nj_get_nuisance_param_name(mod, data, j),
        nj_nuis_param_get(mod, data, j));
    fprintf(logf, "\n");
  }

  if (!silent) fprintf(stderr, "Converged in %d iterations; ELBO=%.2f...\n", t, bestelb);

  vec_free(model_grad); vec_free(model_natgrad); vec_free(sigma_kldgrad);
  vec_free(sigma_penalty_grad); vec_free(m_sigma);
  vec_free(m_sigma_prev); vec_free(v_sigma); vec_free(v_sigma_prev); vec_free(best_sigmapar);
  for (k = 0; k < ncomponents; k++) {
    vec_free(m_mu[k]);
    vec_free(v_mu[k]);
    vec_free(best_mu[k]);
    vec_free(mu_kldgrad[k]);
  }
  sfree(m_mu);
  sfree(v_mu);
  sfree(best_mu);
  sfree(mu_kldgrad);
  sfree(component_t);
  sfree(s); sfree(st); sfree(sd); sfree(sm);
  vec_free(center);
  
  if (n_nuisance_params > 0) {
    vec_free(ave_nuis_grad); vec_free(m_nuis); vec_free(v_nuis);
    vec_free(m_nuis_prev); vec_free(v_nuis_prev); vec_free(best_nuis_params);
  }    
}

/* estimate key components of the ELBO by Monte Carlo integration,
   over a minibatch of size nminibatch.  Returns the expected log
   likelihood.  The model_grad, ave_nuis_grad, ave_lprior, and avemigll
   parameters are updated.  For mixture models, kld is also updated 
   if not NULL. */ 
static void nj_lowr_map_std(multi_MVN *mmvn, Vector *points_std,
                            Vector *points) {
  int k = mmvn->mvn->lowR->ncols;
  Vector *xcomp = vec_new(mmvn->n), *stdproj = vec_new(k);

  for (int d = 0; d < mmvn->d; d++) {
    mmvn_project_down(mmvn, points_std, stdproj, d);
    mmvn->mvn->mu = mmvn->mu[d];
    mvn_map_std(mmvn->mvn, xcomp, stdproj);
    mmvn_project_up(mmvn, xcomp, points, d);
  }

  vec_free(xcomp);
  vec_free(stdproj);
}

/* Return one sample's mixture KLD contribution.  If sigma_kldgrad is non-NULL,
   also accumulate the corresponding -KLD gradients.  Mean gradients go to
   mu_kldgrad by component; shared covariance gradients go to
   sigma_kldgrad's sigma block. */
static double nj_mix_kld_sample_grad(mixture_MVN *mixmvn, int component,
                                     CovarData *data, Vector *points,
                                     Vector *points_std, Vector *sigma_kldgrad,
                                     Vector **mu_kldgrad) {
  multi_MVN *mmvn = mixmvn_get_component(mixmvn, component);
  int fulld = data->nseqs * data->dim;
  double max_ldens = -INFINITY, sum_exp = 0.0, log_qmix, retval;
  double *ldens = smalloc(mixmvn->ncomponents * sizeof(double));
  double *resp = sigma_kldgrad != NULL ?
    smalloc(mixmvn->ncomponents * sizeof(double)) : NULL;

  for (int c = 0; c < mixmvn->ncomponents; c++) {
    ldens[c] = mmvn_log_dens(mixmvn_get_component(mixmvn, c), points);
    if (ldens[c] > max_ldens)
      max_ldens = ldens[c];
  }
  for (int c = 0; c < mixmvn->ncomponents; c++)
    sum_exp += exp(ldens[c] - max_ldens);
  log_qmix = max_ldens + log(sum_exp) - log((double)mixmvn->ncomponents);

  retval = log_qmix;
  if (data->treeprior == NULL)
    retval += 0.5 * (fulld * log(2 * M_PI) +
                     vec_inner_prod(points, points));

  if (sigma_kldgrad != NULL) {
    for (int c = 0; c < mixmvn->ncomponents; c++)
      resp[c] = exp(ldens[c] - max_ldens) / sum_exp;

    if (data->type != LOWR) {
      Vector *kld_dL_dx = vec_new(fulld);
      Vector **prec_resid = smalloc(mixmvn->ncomponents * sizeof(Vector*));

      /* Pathwise part: derivative wrt the sampled point x, then through
         x = mu_component + Sigma_component^(1/2) z. */
      vec_zero(kld_dL_dx);
      for (int c = 0; c < mixmvn->ncomponents; c++) {
        multi_MVN *comp = mixmvn_get_component(mixmvn, c);
        prec_resid[c] = vec_new(fulld);
        vec_zero(prec_resid[c]);

        if (data->type == CONST || data->type == DIAG) {
          for (int pidx = 0; pidx < fulld; pidx++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(comp, pidx);
            double var = data->type == CONST ? data->lambda :
              mat_get(comp->mvn->sigma, pidx, pidx);
            vec_set(prec_resid[c], pidx, centered / var);
          }
        }
        else { /* DIST: use the eigendecomposition of the shared covariance */
          for (int d = 0; d < comp->d; d++) {
            for (int eig = 0; eig < comp->n; eig++) {
              double coeff = 0.0;
              for (int taxon = 0; taxon < comp->n; taxon++) {
                int pidx = taxon * comp->d + d;
                double centered = vec_get(points, pidx) -
                  mmvn_get_mu_el(comp, pidx);
                coeff += mat_get(comp->mvn->evecs, taxon, eig) * centered;
              }
              coeff /= vec_get(comp->mvn->evals, eig);
              for (int taxon = 0; taxon < comp->n; taxon++) {
                int pidx = taxon * comp->d + d;
                vec_set(prec_resid[c], pidx,
                        vec_get(prec_resid[c], pidx) +
                        mat_get(comp->mvn->evecs, taxon, eig) * coeff);
              }
            }
          }
        }

        for (int pidx = 0; pidx < fulld; pidx++)
          vec_set(kld_dL_dx, pidx, vec_get(kld_dL_dx, pidx) +
                  resp[c] * vec_get(prec_resid[c], pidx));
      }
      if (data->treeprior == NULL) {
        for (int pidx = 0; pidx < fulld; pidx++)
          vec_set(kld_dL_dx, pidx, vec_get(kld_dL_dx, pidx) -
                  vec_get(points, pidx));
      }

      /* All mixture mean terms, including the selected component's pathwise
         term through x(mu), live in the per-component mean-gradient vectors. */
      if (mu_kldgrad != NULL) {
        for (int pidx = 0; pidx < fulld; pidx++)
          vec_set(mu_kldgrad[component], pidx,
                  vec_get(mu_kldgrad[component], pidx) +
                  vec_get(kld_dL_dx, pidx));
        for (int c = 0; c < mixmvn->ncomponents; c++)
          for (int pidx = 0; pidx < fulld; pidx++)
            vec_set(mu_kldgrad[c], pidx,
                    vec_get(mu_kldgrad[c], pidx) -
                    resp[c] * vec_get(prec_resid[c], pidx));
      }

      /* Explicit covariance derivative of log q_mix plus the pathwise
         covariance derivative through the selected component sample. */
      if (data->type == DIAG) {
        for (int pidx = 0; pidx < fulld; pidx++) {
          double selected_centered = vec_get(points, pidx) -
            mmvn_get_mu_el(mmvn, pidx);
          double explicit_sigma = 0.0;
          for (int c = 0; c < mixmvn->ncomponents; c++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(mixmvn_get_component(mixmvn, c), pidx);
            explicit_sigma += 0.5 * resp[c] *
              (1.0 - centered * vec_get(prec_resid[c], pidx));
          }
          vec_set(sigma_kldgrad, pidx,
                  vec_get(sigma_kldgrad, pidx) +
                  0.5 * vec_get(kld_dL_dx, pidx) * selected_centered +
                  explicit_sigma);
        }
      }
      else {
        double loglambda_grad = 0.0, explicit_sigma = 0.0;
        for (int c = 0; c < mixmvn->ncomponents; c++) {
          double quad = 0.0;
          for (int pidx = 0; pidx < fulld; pidx++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(mixmvn_get_component(mixmvn, c), pidx);
            quad += centered * vec_get(prec_resid[c], pidx);
          }
          explicit_sigma += 0.5 * resp[c] * (fulld - quad);
        }
        for (int pidx = 0; pidx < fulld; pidx++) {
          double selected_centered = vec_get(points, pidx) -
            mmvn_get_mu_el(mmvn, pidx);
          loglambda_grad += 0.5 * vec_get(kld_dL_dx, pidx) *
            selected_centered;
        }
        vec_set(sigma_kldgrad, 0, vec_get(sigma_kldgrad, 0) +
                loglambda_grad + explicit_sigma);
      }

      for (int c = 0; c < mixmvn->ncomponents; c++)
        vec_free(prec_resid[c]);
      free(prec_resid);
      vec_free(kld_dL_dx);
    }
    else {
      Vector *points_tweak = vec_new(fulld);

      /* LOWR fallback: finite-difference the terms involving R.  This keeps
         the tricky projected-density derivative isolated from the MC loop. */
      for (int pidx = 0; pidx < fulld; pidx++) {
        double orig_x = vec_get(points, pidx);
        double fplus, fminus;

        vec_copy(points_tweak, points);
        vec_set(points_tweak, pidx, orig_x + DERIV_EPS);
        fplus = mixmvn_log_dens(mixmvn, points_tweak);
        if (data->treeprior == NULL)
          fplus += 0.5 * (fulld * log(2 * M_PI) +
                          vec_inner_prod(points_tweak, points_tweak));

        vec_copy(points_tweak, points);
        vec_set(points_tweak, pidx, orig_x - DERIV_EPS);
        fminus = mixmvn_log_dens(mixmvn, points_tweak);
        if (data->treeprior == NULL)
          fminus += 0.5 * (fulld * log(2 * M_PI) +
                           vec_inner_prod(points_tweak, points_tweak));

        if (mu_kldgrad != NULL)
          vec_set(mu_kldgrad[component], pidx,
                  vec_get(mu_kldgrad[component], pidx) -
                  (fplus - fminus) / (2.0 * DERIV_EPS));
      }
      if (mu_kldgrad != NULL) {
        for (int c = 0; c < mixmvn->ncomponents; c++) {
          multi_MVN *comp = mixmvn_get_component(mixmvn, c);
          for (int pidx = 0; pidx < fulld; pidx++) {
            double orig_mu = mmvn_get_mu_el(comp, pidx);
            double fplus, fminus;

            mmvn_set_mu_el(comp, pidx, orig_mu + DERIV_EPS);
            fplus = mixmvn_log_dens(mixmvn, points);
            if (data->treeprior == NULL)
              fplus += 0.5 * (fulld * log(2 * M_PI) +
                              vec_inner_prod(points, points));

            mmvn_set_mu_el(comp, pidx, orig_mu - DERIV_EPS);
            fminus = mixmvn_log_dens(mixmvn, points);
            if (data->treeprior == NULL)
              fminus += 0.5 * (fulld * log(2 * M_PI) +
                               vec_inner_prod(points, points));

            mmvn_set_mu_el(comp, pidx, orig_mu);
            vec_set(mu_kldgrad[c], pidx,
                    vec_get(mu_kldgrad[c], pidx) -
                    (fplus - fminus) / (2.0 * DERIV_EPS));
          }
        }
      }
      for (int pidx = 0; pidx < data->params->size; pidx++) {
        double orig_param = vec_get(data->params, pidx);
        double fplus, fminus;

        vec_set(data->params, pidx, orig_param + DERIV_EPS);
        mixmvn_update_covariance(mixmvn, data);
        nj_lowr_map_std(mmvn, points_std, points_tweak);
        fplus = mixmvn_log_dens(mixmvn, points_tweak);
        if (data->treeprior == NULL)
          fplus += 0.5 * (fulld * log(2 * M_PI) +
                          vec_inner_prod(points_tweak, points_tweak));

        vec_set(data->params, pidx, orig_param - DERIV_EPS);
        mixmvn_update_covariance(mixmvn, data);
        nj_lowr_map_std(mmvn, points_std, points_tweak);
        fminus = mixmvn_log_dens(mixmvn, points_tweak);
        if (data->treeprior == NULL)
          fminus += 0.5 * (fulld * log(2 * M_PI) +
                           vec_inner_prod(points_tweak, points_tweak));

        vec_set(data->params, pidx, orig_param);
        mixmvn_update_covariance(mixmvn, data);
        vec_set(sigma_kldgrad, pidx,
                vec_get(sigma_kldgrad, pidx) -
                (fplus - fminus) / (2.0 * DERIV_EPS));
      }

      vec_free(points_tweak);
    }
  }

  free(ldens);
  if (resp != NULL)
    free(resp);
  return retval;
}

/* Estimate only the mixture KLD term used when Taylor supplies the
   likelihood/prior model gradient.  This avoids the expensive model
   likelihood calculation in nj_elbo_montecarlo while still handling
   E_q[log q_mix(x)], whose mixture entropy has no simple closed form. */
static double nj_elbo_mix_kld_montecarlo(mixture_MVN *mixmvn, int component,
                                         CovarData *data, int nminibatch,
                                         double *kld,
                                         Vector *sigma_kldgrad,
                                         Vector **mu_kldgrad) {
  multi_MVN *mmvn = mixmvn_get_component(mixmvn, component);
  int fulld = data->nseqs * data->dim;
  Vector *points = vec_new(fulld), *points_std;
  double kld_sum = 0.0;
  double scale = data->kld_upweight /
    (nminibatch * data->pointscale * data->pointscale);

  if (data->type == LOWR)
    points_std = vec_new(data->lowrank * data->dim);
  else
    points_std = vec_new(fulld);

  vec_zero(sigma_kldgrad);
  if (mu_kldgrad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_zero(mu_kldgrad[c]);

  for (int i = 0; i < nminibatch; i++) {
    nj_sample_points(mmvn, points, points_std);
    kld_sum += nj_mix_kld_sample_grad(mixmvn, component, data, points,
                                      points_std, sigma_kldgrad,
                                      mu_kldgrad);
  }

  *kld = kld_sum * scale;
  vec_scale(sigma_kldgrad, scale);
  if (mu_kldgrad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_scale(mu_kldgrad[c], scale);

  vec_free(points);
  vec_free(points_std);
  return *kld;
}

double nj_elbo_montecarlo(TreeModel *mod, mixture_MVN *mixmvn, int component,
                          CovarData *data,
                          int nminibatch, Vector *model_grad, Vector *ave_nuis_grad,
                          double *ave_lprior, double *avemigll, double *kld,
                          Vector *sigma_kldgrad, Vector **mu_kldgrad) {
  Vector *grad = vec_new(model_grad->size), *nuis_grad = NULL, *points, *points_std;
  double ll, migll = 0, lprior = 0, avell = 0;
  int n = data->nseqs, dim = data->dim, fulld = n*dim;
  multi_MVN *mmvn = mixmvn_get_component(mixmvn, component);
  int estimate_kld = (mixmvn->ncomponents > 1 && kld != NULL);
  int estimate_kld_grad = (estimate_kld && sigma_kldgrad != NULL);
  double scale = data->kld_upweight /
    (nminibatch * data->pointscale * data->pointscale);

  vec_zero(model_grad);
  if (estimate_kld_grad) {
    vec_zero(sigma_kldgrad);
    if (mu_kldgrad != NULL)
      for (int c = 0; c < mixmvn->ncomponents; c++)
        vec_zero(mu_kldgrad[c]);
  }
  if (ave_nuis_grad != NULL) {
    nuis_grad = vec_new(ave_nuis_grad->size);
    vec_zero(ave_nuis_grad);
  }

  *ave_lprior = *avemigll = 0;
  if (estimate_kld)
    *kld = 0;

  points = vec_new(fulld);
  if (data->type == LOWR) /* in this case, the underlying standard
                             normal MVN is of the lower dimension */
    points_std = vec_new(data->lowrank * dim);
  else
    points_std = vec_new(fulld);

  for (int i = 0; i < nminibatch; i++) {
    migll = 0;
    lprior = 0;
    vec_zero(grad);

    nj_sample_points(mmvn, points, points_std);
    /* Prior contribution to grad is routed inside nj_compute_model_grad
       via dL_dt -> Jacobian -> dL_dx; the prior's nuisance grads
       (relclock_sig_grad, nodetimes_grad) are then picked up below by
       nj_update_nuis_grad. */
    ll = nj_compute_model_grad(mod, mmvn, points, points_std, grad, data,
                               component, NULL, &migll, &lprior);
    assert(isfinite(ll));

    avell += ll;
    (*avemigll) += migll;
    (*ave_lprior) += lprior;
    if (estimate_kld)
      *kld += nj_mix_kld_sample_grad(mixmvn, component, data, points,
                                     points_std,
                                     estimate_kld_grad ? sigma_kldgrad : NULL,
                                     estimate_kld_grad ? mu_kldgrad : NULL);
    vec_plus_eq(model_grad, grad);

    if (ave_nuis_grad != NULL) {
      vec_zero(nuis_grad);
      nj_update_nuis_grad(mod, data, nuis_grad);
      vec_plus_eq(ave_nuis_grad, nuis_grad);
    }
  }

  /* divide by nminibatch to get expected gradient */
  vec_scale(model_grad, 1.0/nminibatch);
  avell /= nminibatch;
  (*ave_lprior) /= nminibatch;
  (*avemigll) /= nminibatch;
  if (estimate_kld)
    *kld *= scale;
  if (estimate_kld_grad) {
    vec_scale(sigma_kldgrad, scale);
    if (mu_kldgrad != NULL)
      for (int c = 0; c < mixmvn->ncomponents; c++)
        vec_scale(mu_kldgrad[c], scale);
  }

  /* same for nuisance grad if needed */
  if (ave_nuis_grad != NULL)
    vec_scale(ave_nuis_grad, 1.0 / nminibatch);

  /* free everything and return */
  vec_free(points); vec_free(points_std); vec_free(grad);
  if (ave_nuis_grad != NULL)
    vec_free(nuis_grad);

  /* we also have to free the last tree in the tree model to avoid a
     memory leak */
  tr_free(mod->tree);
  mod->tree = NULL;
  
  return avell;
}

/* sample a list of trees from the approximate posterior distribution
   and return as a new list.  If logdens is non-null, return
   corresponding vector of log densities for the samples */
List *nj_var_sample(int nsamples, mixture_MVN *mixmvn, CovarData *data, char** names,
                    Vector *logdens) {
  List *retval = lst_new_ptr(nsamples);
  int i, component;
  multi_MVN *mmvn;
  TreeNode *tree;
  Vector *points_x = vec_new(data->dim * data->nseqs),
    *points_y = vec_new(data->dim * data->nseqs);
  
  for (i = 0; i < nsamples; i++) {
    component = mixmvn_sample_component(mixmvn);
    mmvn = mixmvn_get_component(mixmvn, component);
    nj_sample_points(mmvn, points_x, NULL);
    
    if (logdens != NULL) 
      vec_set(logdens, i, mixmvn_log_dens(mixmvn, points_x));
     
    nj_apply_normalizing_flows(points_y, points_x, data, component, NULL);
    nj_points_to_distances(points_y, data);
    tree = nj_inf(data->dist, names, NULL, NULL, data);
    lst_push_ptr(retval, tree);
  }
  
  vec_free(points_x);
  vec_free(points_y);
  return(retval);
}

/* return a single tree representing the approximate posterior mean */
TreeNode *nj_mean(Vector *mu, char **names, CovarData *data) {
  TreeNode *tree;
  
  if (data->nseqs * data->dim != mu->size)
    die("ERROR in nj_mean: bad dimensions\n");

  nj_points_to_distances(mu, data);  
  tree = nj_inf(data->dist, names, NULL, NULL, data);
  
  return(tree);
}

/* sample points from variational distribution.  This is a wrapper
   that encapsulates the use of antithetic sampling.  If points_std is
   non-NULL, it will be used to store the baseline standard normal
   variate for use in downstream calculations in variational
   inference. Antithetic sampling is only used in this case */
void nj_sample_points(multi_MVN *mmvn, Vector *points, Vector *points_std) {
  static int i = 0;
  static Vector *cachedpoints = NULL, *cachedstd = NULL;  
      
  if (points_std == NULL) 
    mmvn_sample(mmvn, points); /* simple in this case */
  else {
    /* otherwise we have to make use of caching for antithetic sampling */
    if (cachedpoints != NULL && cachedpoints->size != points->size) {
      vec_free(cachedpoints);
      vec_free(cachedstd);
      cachedpoints = NULL; /* force realloc */
    }
    if (cachedpoints == NULL) {
      cachedpoints = vec_new(points->size);
      cachedstd = vec_new(points_std->size);   
      i = 0; /* force new sample */
    }
    
    if (i % 2 == 0) { /* new sample, update caches */
      mmvn_sample_anti_keep(mmvn, points, cachedpoints, points_std);
      vec_copy(cachedstd, points_std);

    }
    else { /* just use cache to define sample */
      vec_copy(points, cachedpoints);
      vec_copy(points_std, cachedstd);
      vec_scale(points_std, -1.0);
    }
    i++;
  }
}

/* given points_x, apply normalizing flows to compute points_y as y =
   f(x).  Optionally populates *logdet with total log determinate of
   Jacobian (if non-NULL) */
void nj_apply_normalizing_flows(Vector *points_y, Vector *points_x,
                                CovarData *data, int component,
                                double *logdet) {
  double ldet = 0;
  assert(points_x->size == points_y->size);
  assert(component >= 0 && component < data->nflow_components);
  
  if (data->rfs[component] == NULL && data->pfs[component] == NULL) {
    if (logdet != NULL) *logdet = 0;
    vec_copy(points_y, points_x);
    return;
  }

  if (data->rfs[component] != NULL && data->pfs[component] != NULL) {
    /* in this case we need an intermediate vector */
    Vector *tmp = vec_new(points_x->size);
    ldet = rf_forward(data->rfs[component], tmp, points_x);
    ldet += pf_forward(data->pfs[component], points_y, tmp);
    vec_free(tmp);
  }
 
  else if (data->rfs[component] != NULL)
    ldet = rf_forward(data->rfs[component], points_y, points_x);

  else if (data->pfs[component] != NULL)
    ldet = pf_forward(data->pfs[component], points_y, points_x);

  if (logdet != NULL)
    (*logdet) = ldet; 
}

/* compute partial derivatives of KLD wrt variance parameters in LOWR
   case */
void nj_set_kld_sigma_grad_LOWR(Vector *sigma_kldgrad, multi_MVN *mmvn) {
  int i, j;
  Matrix *Rgrad = mat_new(mmvn->mvn->lowR->nrows, mmvn->mvn->lowR->ncols);

  /* calculate partial derivatives using matrix operations, making use
     of precomputed R^T x R */
  mat_mult(Rgrad, mmvn->mvn->lowR, mmvn->mvn->lowR_invRtR);
  mat_minus_eq(Rgrad, mmvn->mvn->lowR);
  mat_scale(Rgrad, mmvn->d);  /* note: computing negative gradient; that is what we need */

  /* populate vector from matrix */
  for (i = 0; i < mmvn->mvn->lowR->nrows; i++) 
    for (j = 0; j < mmvn->mvn->lowR->ncols; j++) 
      vec_set(sigma_kldgrad, i*mmvn->mvn->lowR->ncols + j,
              mat_get(Rgrad, i, j));

  mat_free(Rgrad);
}

/* compute partial derivatives of entropy H[q(x)] wrt LOWR variance
   parameters: Sigma_0 = I + R R^T, Sigma = I_d ⊗ Sigma_0. */
void nj_set_entropy_sigma_grad_LOWR(Vector *sigma_entgrad, multi_MVN *mmvn) {
  int i, j;
  Matrix *Rgrad = mat_new(mmvn->mvn->lowR->nrows, mmvn->mvn->lowR->ncols);

  /* For entropy, only the log det term contributes:
       H[q] = (d/2) * log det(Sigma_0) + const
     For Sigma_0 = I + R R^T, using the matrix determinant lemma,
       ∂H/∂R = d * R * (I + R^T R)^{-1}
     and lowR_invRtR is precomputed as (I + R^T R)^{-1}.
  */
  mat_mult(Rgrad, mmvn->mvn->lowR, mmvn->mvn->lowR_invRtR);
  mat_scale(Rgrad, mmvn->d);   /* computing positive gradient of +H[q] */

  /* populate vector from matrix */
  for (i = 0; i < mmvn->mvn->lowR->nrows; i++)
    for (j = 0; j < mmvn->mvn->lowR->ncols; j++)
      vec_set(sigma_entgrad,
              i*mmvn->mvn->lowR->ncols + j,
              mat_get(Rgrad, i, j));

  mat_free(Rgrad);
}
