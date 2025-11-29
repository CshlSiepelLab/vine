/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
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
#include <geometry.h>
#include <adam_scheduler.h>
#include <sparse_matrix.h>
#include <gradients.h>
#include <nuisance.h>

/* uncomment to dump gradients to a file called "grads_log.txt" */
//#define DUMPGRAD 1

/* optimize variational model by stochastic gradient ascent using the
   Adam algorithm.  Takes initial tree model and alignment and
   distance matrix, dimensionality of Euclidean space to work in.
   Note: alters distance matrix */
void nj_variational_inf(TreeModel *mod, multi_MVN *mmvn,
                        int nminibatch, double learnrate, int nbatches_conv,
                        int min_nbatches, CovarData *data, FILE *logf) {

  Vector *points, *grad, *kldgrad, *avegrad, *m, *m_prev, *v, *v_prev,
    *best_mu, *best_sigmapar, *rescaledgrad, *sparsitygrad = NULL, 
    *points_std;
  Vector *sigmapar = data->params;
  int n = data->nseqs, i, j, t, stop = FALSE, bestt = -1, graddim,
    dim = data->dim, fulld = n*dim;
  double elb, ll, avell, migll, avemigll, kld, bestelb = -INFTY, bestll = -INFTY,
    bestkld = -INFTY, bestmigll = -INFTY,
    running_tot = 0, last_running_tot = -INFTY, trace, logdet, penalty = 0,
    bestpenalty = 0, ave_nf_logdet, best_nf_logdet = -INFTY,
    ave_lprior, best_lprior = -INFTY, subsamp_rescale = 1.0;
  FILE *gradf = NULL;

  /* for nuisance parameters; these are parameters that are optimized
     by stochastic gradient descent but are not fully sampled via the
     variational distribution */
  int n_nuisance_params = nj_get_num_nuisance_params(mod, data);
  Vector *nuis_grad = NULL, *ave_nuis_grad = NULL, *m_nuis = NULL,
    *v_nuis = NULL, *m_nuis_prev = NULL, *v_nuis_prev = NULL,
    *best_nuis_params = NULL, *prior_grad = NULL;
  
#ifdef DUMPGRAD
  gradf = phast_fopen("grads_log.txt", "w");
#endif
  
  if (mmvn->d * mmvn->n != dim * n)
    die("ERROR in nj_variational_inf: bad dimensions\n");

  points = vec_new(fulld);
  graddim = fulld + data->params->size;
  grad = vec_new(graddim);  
  kldgrad = vec_new(graddim);
  avegrad = vec_new(graddim);
  rescaledgrad = vec_new(graddim);
  m = vec_new(graddim);
  m_prev = vec_new(graddim);
  v = vec_new(graddim);
  v_prev = vec_new(graddim);
  sparsitygrad = vec_new(graddim);

  if (n_nuisance_params > 0) {
    nuis_grad = vec_new(n_nuisance_params);
    ave_nuis_grad = vec_new(n_nuisance_params);
    m_nuis = vec_new(n_nuisance_params);
    v_nuis = vec_new(n_nuisance_params);
    m_nuis_prev = vec_new(n_nuisance_params);
    v_nuis_prev = vec_new(n_nuisance_params);
    best_nuis_params = vec_new(n_nuisance_params);
  }

  if (data->treeprior != NULL)
    prior_grad = vec_new(graddim);  

  if (data->type == LOWR) /* in this case, the underlying standard
                             normal MVN is of the lower dimension */
    points_std = vec_new(data->lowrank * dim);
  else
    points_std = vec_new(fulld);
  
  best_mu = vec_new(fulld);
  mmvn_save_mu(mmvn, best_mu);
  best_sigmapar = vec_new(sigmapar->size);
  vec_copy(best_sigmapar, sigmapar);

  /* set up log file */
  if (logf != NULL) {
    fprintf(logf, "# VINE logfile\n");
    fprintf(logf, "state\tll\telbo\t");
    if (data->treeprior != NULL)
      fprintf(logf, "prior\t");
    else
      fprintf(logf, "kld\t");
    if (data->var_reg != 0)
      fprintf(logf, "penalty\t");
    if (data->rf != NULL || data->pf != NULL)
      fprintf(logf, "nfld\t");
    if (data->crispr_mod == NULL)
      fprintf(logf, "subsamp\treuse\tgradnorm\tclip\t");
    if (data->migtable != NULL)
      fprintf(logf, "mig_ll\t");
    for (j = 0; j < fulld; j++)
      fprintf(logf, "mu.%d\t", j);
    for (j = 0; j < sigmapar->size; j++)
      fprintf(logf, "sigma.%d\t", j);
    for (j = 0; j < n_nuisance_params; j++)
      fprintf(logf, "%s\t", nj_get_nuisance_param_name(mod, data, j));
    fprintf(logf, "\n");
  }
  if (gradf != NULL) {
    fprintf(gradf, "state\t");
    for (j = 0; j < fulld; j++)
      fprintf(gradf, "g_mu.%d\t", j);
    for (j = fulld; j < grad->size; j++)
      fprintf(gradf, "g_sig.%d\t", j-fulld);
    fprintf(gradf, "\n");
  }
  
  /* initialize moments for Adam algorithm */
  vec_zero(m);  vec_zero(m_prev);
  vec_zero(v);  vec_zero(v_prev);
  if (n_nuisance_params > 0) {
    vec_zero(m_nuis);  vec_zero(m_nuis_prev);
    vec_zero(v_nuis);  vec_zero(v_nuis_prev);
  }
  t = 0;

  /* set up scheduler; initialized but not currently used in CRISPR mode */
  int maxlen = data->crispr_mod == NULL ? data->msa->length : 1000;
  Scheduler *s = sched_new(maxlen, NSUBSAMPLES, 20,
                           learnrate, 10, 50, 30);
  SchedState *st = sched_new_state(s);
  SchedDirectives *sd = smalloc(sizeof(SchedDirectives));
  SchedMetrics *sm = smalloc(sizeof(SchedMetrics));
  sm->grad_norm = 0;
  
  do {
    /* get directives from scheduler */
    sched_next(s, st, sm, sd);
    unsigned int clipped = FALSE;
    
    /* we can precompute the KLD because it does not depend on the data under this model */
    /* (see equation 7, Doersch arXiv 2016) */
    kld = 0;
    vec_zero(kldgrad);
    if (data->treeprior == NULL) { /* only do if no explicit tree prior */
      trace = mmvn_trace(mmvn);  /* we'll reuse this */
      logdet = mmvn_log_det(mmvn);
    
      kld = 0.5 * (trace + mmvn_mu2(mmvn) - fulld - logdet);

      kld *= data->kld_upweight/(data->pointscale*data->pointscale);      
    
      /* we can also precompute the contribution of the KLD to the gradient */
      /* Note KLD is subtracted rather than added, so compute the gradient of -KLD */
      for (j = 0; j < grad->size; j++) {
        double gj = 0.0;

        if (j < n*dim)  /* partial deriv wrt mu_j is just mu_j */
          gj = -1.0*mmvn_get_mu_el(mmvn, j);
        else {            /* partial deriv wrt sigma_j is more
                             complicated because of the trace and log
                             determinant */
          if (data->type == CONST || data->type == DIST)
            gj = 0.5 * (fulld - trace);
          else if (data->type == DIAG) 
            gj = 0.5 * (1.0 - mat_get(mmvn->mvn->sigma, j-fulld, j-fulld)); 
          else 
            continue; /* LOWR case is messy; handle below */
        }
        vec_set(kldgrad, j, gj);
      }
    
      if (data->type == LOWR) 
        nj_set_kld_grad_LOWR(kldgrad, mmvn);
    }
    
    /* can also pre-compute variance penalty */
    vec_zero(sparsitygrad);
    nj_compute_variance_penalty(sparsitygrad, mmvn, data);
    penalty = data->var_pen;

    vec_scale(kldgrad, data->kld_upweight/(data->pointscale*data->pointscale));
    
    /* now sample a minibatch from the MVN averaging distribution and
       compute log likelihoods and gradients */
    vec_zero(avegrad);
    if (n_nuisance_params > 0)
      vec_zero(ave_nuis_grad);
    if (data->treeprior != NULL)
      vec_zero(prior_grad);
    avell = 0;
    ave_lprior = 0;
    ave_nf_logdet = 0;
    avemigll = 0;

    /* set up subsampling for this minibatch (but not in crispr mode) */
    if (!sd->full_grad_now && data->crispr_mod == NULL) {
      data->subsample = TRUE;
      data->subsampsize = sd->m;
      data->reuse_subsamp = !sd->resample_sites;
      subsamp_rescale = (double)data->msa->length / data->subsampsize;
    }
    else { /* no subsampling */
      data->subsample = FALSE;
      subsamp_rescale = 1.0;
    }

    for (i = 0; i < nminibatch; i++) {
      int bail = 0;
      double nf_logdet = 0; /* log det of Jacobian for normalizing flows;
                               will only be set if needed */
      
      do {
        /* do this in a way that keeps track of the original standard
           normal variable (points_std) for use in computing
           gradients.  Also use antithetic sampling to reduce
           variance */
        nj_sample_points(mmvn, points, points_std);
 
        ll = nj_compute_model_grad(mod, mmvn, points, points_std, grad, data, &nf_logdet, &migll);
       
        if (++bail > 10 && !isfinite(ll)) {
          fprintf(stderr, "WARNING: repeatedly sampling zero-probability trees. Prohibiting zero-length branches.\n");
          data->no_zero_br = TRUE;
          assert(bail < 15); /* prohibit infinite loop */
        }
      } while (!isfinite(ll));  /* in certain cases under the
                                   irreversible CRISPR model, trees
                                   can have likelihoods of zero; we'll
                                   try to just keep sampling if that
                                   happens and if necessary constrain
                                   the tree structure */

      if (data->subsample == TRUE)  /* rescale ll if subsampling */
        ll *= subsamp_rescale;

      avell += ll;
      avemigll += migll;
      vec_plus_eq(avegrad, grad);
      ave_nf_logdet += nf_logdet;

      /* calculate prior if needed; add gradient of branches */
      if (data->treeprior != NULL) {
        ave_lprior += tp_compute_log_prior(mod, data, prior_grad);
        vec_plus_eq(avegrad, prior_grad);
      }

      if (n_nuisance_params > 0) {
        nj_update_nuis_grad(mod, data, nuis_grad);
        vec_plus_eq(ave_nuis_grad, nuis_grad);
      }
    }

    /* divide by nminibatch to get expected gradient */
    vec_scale(avegrad, 1.0/nminibatch);
    avell /= nminibatch;
    ave_lprior /= nminibatch;
    ave_nf_logdet /= nminibatch;
    avemigll /= nminibatch;
    
    vec_plus_eq(avegrad, kldgrad);
    vec_plus_eq(avegrad, sparsitygrad);

    if (n_nuisance_params > 0) 
      vec_scale(ave_nuis_grad, 1.0/nminibatch);
    
    /* store parameters if best yet */
    elb = avell + ave_lprior - kld + ave_nf_logdet - penalty + avemigll;
    if (elb > bestelb && (sd->full_grad_now || data->crispr_mod != NULL)) {
      bestelb = elb;
      bestll = avell;  /* not necessarily best ll but ll corresponding to bestelb */
      best_lprior = ave_lprior;
      bestkld = kld;  
      bestpenalty = penalty;
      best_nf_logdet = ave_nf_logdet;
      bestmigll = avemigll;
      bestt = t;
      mmvn_save_mu(mmvn, best_mu);
      vec_copy(best_sigmapar, sigmapar);
      if (n_nuisance_params > 0)
        nj_save_nuis_params(best_nuis_params, mod, data);
    }

    /* rescale gradient by approximate inverse Fisher information to
       put on similar scales; seems to help with optimization */
    if (data->natural_grad == TRUE)
      nj_rescale_grad(avegrad, rescaledgrad, mmvn, data);
    else
      vec_copy(rescaledgrad, avegrad);
    /* we won't do this with nuisance params */

    /* update scheduler with norm of gradient and clip if necessary */
    if (data->crispr_mod == NULL) {
      sm->grad_norm = vec_norm(rescaledgrad);
      if (sd->clip_norm > 0 && sm->grad_norm > sd->clip_norm) {
        vec_scale(rescaledgrad, sd->clip_norm / sm->grad_norm);
        clipped = TRUE;
      }
    }

    /* Adam updates; see Kingma & Ba, arxiv 2014 */
    t++;
    for (j = 0; j < rescaledgrad->size; j++) {   
      double mhatj, vhatj, g = vec_get(rescaledgrad, j);
      
      vec_set(m, j, ADAM_BETA1 * vec_get(m_prev, j) + (1.0 - ADAM_BETA1) * g);
      vec_set(v, j, ADAM_BETA2 * vec_get(v_prev, j) + (1.0 - ADAM_BETA2) * pow(g,2));
      mhatj = vec_get(m, j) / (1.0 - pow(ADAM_BETA1, t));
      vhatj = vec_get(v, j) / (1.0 - pow(ADAM_BETA2, t));

      /* update mu or sigma, depending on parameter index */
      if (j < fulld)
        mmvn_set_mu_el(mmvn, j, mmvn_get_mu_el(mmvn, j) +
                               sd->lr * mhatj / (sqrt(vhatj) + ADAM_EPS)); 
      else {
        vec_set(sigmapar, j-fulld, vec_get(sigmapar, j-fulld) +
                sd->lr * mhatj / (sqrt(vhatj) + ADAM_EPS));
      }
    }
    nj_update_covariance(mmvn, data);
    
    vec_copy(m_prev, m);
    vec_copy(v_prev, v);

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
      if (data->var_reg != 0)
        fprintf(logf, "%f\t", data->var_pen);
      if (data->rf != NULL || data->pf != NULL)
        fprintf(logf, "%f\t", ave_nf_logdet);
      if (data->crispr_mod == NULL)
        fprintf(logf, "%d\t%d\t%f\t%d\t", data->subsampsize,
                data->reuse_subsamp, sm->grad_norm, clipped);
      if (data->migtable != NULL) 
        fprintf(logf, "%f\t", avemigll); 
      mmvn_print(mmvn, logf, TRUE, FALSE);
      for (j = 0; j < sigmapar->size; j++)
        fprintf(logf, "%f\t", vec_get(sigmapar, j));
      for (j = 0; j < n_nuisance_params; j++)
        fprintf(logf, "%f\t", nj_nuis_param_get(mod, data, j)); 
      fprintf(logf, "\n");
    }
    if (gradf != NULL) {
      fprintf(gradf, "%d\t", t);
      for (j = 0; j < rescaledgrad->size; j++)
        fprintf(gradf, "%f\t", vec_get(rescaledgrad, j));
      fprintf(gradf, "\n");
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

  mmvn_set_mu(mmvn, best_mu);
  vec_copy(sigmapar, best_sigmapar);
  nj_update_covariance(mmvn, data);
  if (n_nuisance_params > 0)
    nj_update_nuis_params(best_nuis_params, mod, data);
  
  if (logf != NULL) {
    fprintf(logf,
            "# Reverting to parameters from iteration %d; ELB: %.2f, LNL: "
            "%.2f, LPRIOR: %.2f, KLD: %.2f, RFLD: %.2f, penalty: %.2f",
            bestt + 1, bestelb, bestll, best_lprior, bestkld, best_nf_logdet,
            bestpenalty);
    if (data->migtable != NULL)
      fprintf(logf, ", MIGLL: %.2f", bestmigll);
    for (j = 0; j < n_nuisance_params; j++) /* print these also if available */
      fprintf(logf, ", %s: %.4f", nj_get_nuisance_param_name(mod, data, j),
              nj_nuis_param_get(mod, data, j));
    fprintf(logf, "\n");
  }

  vec_free(grad); vec_free(avegrad); vec_free(rescaledgrad); vec_free(kldgrad);
  vec_free(sparsitygrad); vec_free(points); vec_free(points_std); vec_free(m);
  vec_free(m_prev); vec_free(v); vec_free(v_prev); vec_free(best_mu); vec_free(best_sigmapar);
  sfree(s); sfree(st); sfree(sd); sfree(sm);
  
  if (data->treeprior != NULL)
    vec_free(prior_grad);
  
  if (n_nuisance_params > 0) {
    vec_free(nuis_grad); vec_free(ave_nuis_grad); vec_free(m_nuis); vec_free(v_nuis);
    vec_free(m_nuis_prev); vec_free(v_nuis_prev); vec_free(best_nuis_params);
  }    
}

/* sample a list of trees from the approximate posterior distribution
   and return as a new list.  If logdens is non-null, return
   corresponding vector of log densities for the samples */
List *nj_var_sample(int nsamples, multi_MVN *mmvn, CovarData *data, char** names,
                    Vector *logdens) {
  List *retval = lst_new_ptr(nsamples);
  int i;
  TreeNode *tree;
  Vector *points_x = vec_new(mmvn->d * mmvn->n), *points_y = vec_new(mmvn->d * mmvn->n);
  
  for (i = 0; i < nsamples; i++) {
    nj_sample_points(mmvn, points_x, NULL);
    
    if (logdens != NULL) 
      vec_set(logdens, i, mmvn_log_dens(mmvn, points_x));
     
    nj_apply_normalizing_flows(points_y, points_x, data, NULL);
    nj_points_to_distances(points_y, data);
    tree = nj_inf(data->dist, names, NULL, data);
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
  tree = nj_inf(data->dist, names, NULL, data);
  
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
                                CovarData *data, double *logdet) {
  double ldet = 0;
  assert(points_x->size == points_y->size);
  
  if (data->rf == NULL && data->pf == NULL) {
    if (logdet != NULL) *logdet = 0;
    vec_copy(points_y, points_x);
    return;
  }

  if (data->rf != NULL && data->pf != NULL) {
    /* in this case we need an intermediate vector */
    Vector *tmp = vec_new(points_x->size);
    ldet = rf_forward(data->rf, tmp, points_x);
    ldet += pf_forward(data->pf, points_y, tmp);
    vec_free(tmp);
  }
 
  else if (data->rf != NULL) 
    ldet = rf_forward(data->rf, points_y, points_x);

  else if (data->pf != NULL) 
    ldet = pf_forward(data->pf, points_y, points_x);

  if (logdet != NULL)
    (*logdet) = ldet; 
}

/* compute partial derivatives of KLD wrt variance parameters in LOWR
   case */
void nj_set_kld_grad_LOWR(Vector *kldgrad, multi_MVN *mmvn) {
  int i, j;
  int offset = mmvn->d * mmvn->n;
  Matrix *Rgrad = mat_new(mmvn->mvn->lowR->nrows, mmvn->mvn->lowR->ncols);

  /* calculate partial derivatives using matrix operations, making use
     of precomputed R^T x R */
  mat_mult(Rgrad, mmvn->mvn->lowR, mmvn->mvn->lowR_invRtR);
  mat_minus_eq(Rgrad, mmvn->mvn->lowR);
  mat_scale(Rgrad, mmvn->d);  /* note: computing negative gradient; that is what we need */

  /* populate vector from matrix */
  for (i = 0; i < mmvn->mvn->lowR->nrows; i++) 
    for (j = 0; j < mmvn->mvn->lowR->ncols; j++) 
      vec_set(kldgrad, offset + i*mmvn->mvn->lowR->ncols + j, mat_get(Rgrad, i, j));

  mat_free(Rgrad);
}
