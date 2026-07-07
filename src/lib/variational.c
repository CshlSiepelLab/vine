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
#include <gauge_fixing.h>

/* number of warmup iterations to run with migration model disabled;
   allows tree topology to converge before migration inference activates */
#define CPR_MIG_WARMUP_ITERS 150

/* Mild symmetric Dirichlet prior on mixture weights.  This discourages
   premature component collapse without forcing nearly uniform weights. */
#define MIXTURE_WEIGHT_PRIOR_ALPHA 2.0

static Vector *vec_new_zero(int size) {
  Vector *v = vec_new(size);
  vec_zero(v);
  return v;
}

static void log_header(TreeModel *mod, mixture_MVN *mixmvn,
                                      CovarData *data, FILE *logf,
                                      int fulld, int n_nuisance_params,
                                      unsigned int log_all) {
  int j, k, ncomponents = mixmvn->ncomponents;

  if (logf == NULL)
    return;

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
  if (ncomponents > 1)
    for (k = 0; k < ncomponents; k++)
      fprintf(logf, "mixweight.%d\t", k);
  if (log_all) {
    if (ncomponents > 1) {
      for (k = 0; k < ncomponents; k++) {
        fprintf(logf, "component.%d\t", k);
        for (j = 0; j < fulld; j++)
          fprintf(logf, "mu.%d.%d\t", k, j);
      }
    }
    else
      for (j = 0; j < fulld; j++)
        fprintf(logf, "mu.%d\t", j);
    if (data->type == LOWR || data->type == DIAG) {
      for (k = 0; k < ncomponents; k++)
        for (j = 0; j < data->covar_params[k]->size; j++)
          fprintf(logf, "sigma.%d.%d\t", k, j);
    }
  }
  if (data->type == CONST || data->type == DIST) {
    for (k = 0; k < ncomponents; k++)
      for (j = 0; j < data->covar_params[k]->size; j++)
        fprintf(logf, "sigma.%d.%d\t", k, j);
  }
  for (j = 0; j < n_nuisance_params; j++)
    fprintf(logf, "%s\t", get_nuisance_param_name(mod, data, j));
  fprintf(logf, "\n");
}

static void log_iteration(TreeModel *mod, mixture_MVN *mixmvn,
                                         CovarData *data, FILE *logf,
                                         TaylorData *taylor_stash,
                                         SchedMetrics *sm, int t,
                                         double avell, double elb,
                                         double ave_lprior, double kld,
                                         double penalty,
                                         double avemigll,
                                         unsigned int clipped,
                                         int n_nuisance_params,
                                         unsigned int log_all) {
  int j, k, ncomponents = mixmvn->ncomponents;

  if (logf == NULL)
    return;

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
    fprintf(logf, "%f\t", penalty);
  if (data->crispr_mod == NULL)
    fprintf(logf, "%d\t%d\t%f\t%d\t", data->subsampsize,
            data->reuse_subsamp, sm->grad_norm, clipped);
  if (data->migtable != NULL)
    fprintf(logf, "%f\t", avemigll);
  if (ncomponents > 1)
    for (k = 0; k < ncomponents; k++)
      fprintf(logf, "%f\t", vec_get(mixmvn->weights, k));
  if (log_all) {
    mixmvn_print(mixmvn, logf, TRUE, FALSE);
    if (data->type == LOWR || data->type == DIAG) {
      for (k = 0; k < ncomponents; k++)
        for (j = 0; j < data->covar_params[k]->size; j++)
          fprintf(logf, "%f\t", vec_get(data->covar_params[k], j));
    }
  }
  if (data->type == CONST || data->type == DIST) {
    for (k = 0; k < ncomponents; k++)
      for (j = 0; j < data->covar_params[k]->size; j++)
        fprintf(logf, "%f\t", vec_get(data->covar_params[k], j));
  }
  for (j = 0; j < n_nuisance_params; j++)
    fprintf(logf, "%f\t", nuis_param_get(mod, data, j));

  fprintf(logf, "\n");
}

static void log_running_elbo(FILE *logf, int nbatches_conv,
                                            double running_tot) {
  if (logf != NULL)
    fprintf(logf, "# Average ELBO for last %d: %f\n",
            nbatches_conv, running_tot/nbatches_conv);
}

static void log_final(TreeModel *mod, mixture_MVN *mixmvn,
                                     CovarData *data, FILE *logf,
                                     int bestt, double bestelb,
                                     double bestll, double best_lprior,
                                     double bestkld, double bestpenalty,
                                     double bestll_at_mean,
                                     double final_mc_ll,
                                     double bestmigll,
                                     int n_nuisance_params) {
  int j, ncomponents = mixmvn->ncomponents;

  if (logf == NULL)
    return;

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
  if (ncomponents > 1) {
    fprintf(logf, ", mixweights:");
    for (j = 0; j < ncomponents; j++)
      fprintf(logf, " %.4f", vec_get(mixmvn->weights, j));
  }
  for (j = 0; j < n_nuisance_params; j++) /* print these also if available */
    fprintf(logf, ", %s: %.4f", get_nuisance_param_name(mod, data, j),
      nuis_param_get(mod, data, j));
  fprintf(logf, "\n");
}

/* Closed-form KL(q || N(0, I)) for a single MVN component, plus gradients of
   -KL because the optimizer ascends the ELBO.  The returned scalar is the
   unscaled positive KL term that will be subtracted from the ELBO. */
static double compute_standard_normal_kl_term_and_grads(multi_MVN *component,
                                                           CovarData *data,
                                                           int fulld,
                                                           double logdet,
                                                           Vector *sigma_grad,
                                                           Vector *mu_grad) {
  int j;
  double trace = mmvn_trace(component);

  if (mu_grad != NULL)
    for (j = 0; j < fulld; j++)
      vec_set(mu_grad, j, -1.0 * mmvn_get_mu_el(component, j));

  if (sigma_grad != NULL) {
    for (j = 0; j < sigma_grad->size; j++) {
      double gj = 0.0;

      /* The covariance derivative includes the trace and log determinant terms.
         LOWR uses a matrix expression below because its parameters interact. */
      if (data->type == CONST || data->type == DIST)
        gj = 0.5 * (fulld - trace);
      else if (data->type == DIAG)
        gj = 0.5 * (1.0 - mat_get(component->mvn->sigma, j, j));
      else
        continue;
      vec_set(sigma_grad, j, gj);
    }

    if (data->type == LOWR)
      set_kld_sigma_grad_LOWR(sigma_grad, component);
  }

  return 0.5 * (trace + mmvn_mu2(component) - fulld - logdet);
}

/* With an explicit tree prior, the closed-form variational contribution is
   the negative entropy term in the ELBO accounting variable.  Its optimizer
   gradient is +dH/dtheta, again matching the ELBO ascent convention. */
static double compute_entropy_term_and_grads(multi_MVN *component,
                                                CovarData *data,
                                                int fulld,
                                                double logdet,
                                                Vector *sigma_grad) {
  int j;

  if (sigma_grad != NULL) {
    for (j = 0; j < sigma_grad->size; j++) {
      double gj = 0.0;

      /* Mean parameters do not affect entropy. */
      if (data->type == CONST || data->type == DIST)
        gj = 0.5 * fulld;
      else if (data->type == DIAG)
        gj = 0.5;
      else
        continue;
      vec_set(sigma_grad, j, gj);
    }

    if (data->type == LOWR)
      set_entropy_sigma_grad_LOWR(sigma_grad, component);
  }

  return -0.5 * (fulld * (1.0 + log(2 * M_PI)) + logdet);
}

/* Closed-form KLD (and gradients) for the single mixture component case,
   scaled to match the Monte Carlo KLD estimate used when ncomponents > 1. */
static double compute_closed_form_kld(mixture_MVN *mixmvn, CovarData *data,
                                         int fulld, Vector *sigma_kld_param_grad,
                                         Vector *mu_kld_param_grad) {
  double logdet = mmvn_log_det(mixmvn->components[0]);
  double scale = data->kld_upweight / (data->pointscale * data->pointscale);
  double kld;

  if (data->treeprior == NULL)
    kld = compute_standard_normal_kl_term_and_grads(mixmvn->components[0],
                                                       data, fulld, logdet,
                                                       sigma_kld_param_grad, mu_kld_param_grad);
  else
    kld = compute_entropy_term_and_grads(mixmvn->components[0], data,
                                           fulld, logdet, sigma_kld_param_grad);

  kld *= scale;
  if (sigma_kld_param_grad != NULL)
    vec_scale(sigma_kld_param_grad, scale);
  if (mu_kld_param_grad != NULL)
    vec_scale(mu_kld_param_grad, scale);
  return kld;
}

static double add_mixture_weight_dirichlet_prior(mixture_MVN *mixmvn,
                                                       Vector *component_elbo,
                                                       Vector *weight_kld_param_grad,
                                                       Vector *weight_param_grad,
                                                       double mix_elbo) {
  int j;
  double weight_log_prior = 0.0;

  /* log density of a symmetric Dirichlet prior, ignoring constants */
  for (j = 0; j < mixmvn->ncomponents; j++) {
    double wj = vec_get(mixmvn->weights, j);
    weight_log_prior += (MIXTURE_WEIGHT_PRIOR_ALPHA - 1.0) * log(wj);
  }

  /* compute gradients of the log-prior with respect to the mixture weights */
  for (j = 0; j < mixmvn->ncomponents; j++) {
    double wj = vec_get(mixmvn->weights, j);
    vec_set(weight_param_grad, j,
            vec_get(weight_kld_param_grad, j) +
            wj * (vec_get(component_elbo, j) - mix_elbo) +
            (MIXTURE_WEIGHT_PRIOR_ALPHA - 1.0) *
            (1.0 - mixmvn->ncomponents * wj));
  }

  return mix_elbo + weight_log_prior;
}

/* estimate key components of the ELBO by Monte Carlo integration,
   over a minibatch of size nminibatch. */ 
static void lowr_map_std(multi_MVN *mmvn, Vector *points_std,
                            Vector *points) {
  int k = mmvn->mvn->lowR->ncols;
  Vector *xcomp = vec_new(mmvn->n), *stdproj = vec_new(k);

  for (int d = 0; d < mmvn->d; d++) {
    for (int j = 0; j < k; j++)
      vec_set(stdproj, j, vec_get(points_std, j*mmvn->d + d));
    mmvn->mvn->mu = mmvn->mu[d];
    mvn_map_std(mmvn->mvn, xcomp, stdproj);
    mmvn_project_up(mmvn, xcomp, points, d);
  }

  vec_free(xcomp);
  vec_free(stdproj);
}

/* Return one sample's mixture KLD contribution.  If sigma_kld_param_grad is non-NULL,
   also accumulate the corresponding -KLD gradients.  Mean and covariance
   gradients are indexed by component. */
static double mix_kld_sample_grad(mixture_MVN *mixmvn, int component,
                                     CovarData *data, Vector *points,
                                     Vector *points_std, Vector **sigma_kld_param_grad,
                                     Vector **mu_kld_param_grad,
                                     Vector *weight_kld_param_grad) {
  int fulld = data->nseqs * data->dim;
  double max_ldens = -INFINITY, sum_exp = 0.0, log_qmix, retval;
  double *ldens = smalloc(mixmvn->ncomponents * sizeof(double));
  double *resp = (sigma_kld_param_grad != NULL || weight_kld_param_grad != NULL) ?
    smalloc(mixmvn->ncomponents * sizeof(double)) : NULL;

  for (int c = 0; c < mixmvn->ncomponents; c++) {
    ldens[c] = log(vec_get(mixmvn->weights, c)) +
      mmvn_log_dens(mixmvn->components[c], points);
    if (ldens[c] > max_ldens)
      max_ldens = ldens[c];
  }
  for (int c = 0; c < mixmvn->ncomponents; c++)
    sum_exp += exp(ldens[c] - max_ldens);
  log_qmix = max_ldens + log(sum_exp);

  retval = log_qmix;
  if (data->treeprior == NULL)
    retval += 0.5 * (fulld * log(2 * M_PI) +
                     vec_inner_prod(points, points));

  if (resp != NULL) {
    for (int c = 0; c < mixmvn->ncomponents; c++)
      resp[c] = exp(ldens[c] - max_ldens) / sum_exp;
  }

  if (weight_kld_param_grad != NULL) {
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_set(weight_kld_param_grad, c, vec_get(weight_kld_param_grad, c) +
              vec_get(mixmvn->weights, c) - resp[c]);
  }

  if (sigma_kld_param_grad != NULL) {

    if (data->type != LOWR) {
      Vector *kld_dL_dx = vec_new_zero(fulld);
      Vector **prec_resid = smalloc(mixmvn->ncomponents * sizeof(Vector*));

      /* Pathwise part: derivative wrt the sampled point x, then through
         x = mu_component + Sigma_component^(1/2) z. */
      for (int c = 0; c < mixmvn->ncomponents; c++) {
        multi_MVN *comp = mixmvn->components[c];
        prec_resid[c] = vec_new_zero(fulld);

        if (data->type == CONST || data->type == DIAG) {
          for (int pidx = 0; pidx < fulld; pidx++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(comp, pidx);
            double var = mat_get(comp->mvn->sigma, pidx, pidx);
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
      if (mu_kld_param_grad != NULL) {
        for (int pidx = 0; pidx < fulld; pidx++)
          vec_set(mu_kld_param_grad[component], pidx,
                  vec_get(mu_kld_param_grad[component], pidx) +
                  vec_get(kld_dL_dx, pidx));
        for (int c = 0; c < mixmvn->ncomponents; c++)
          for (int pidx = 0; pidx < fulld; pidx++)
            vec_set(mu_kld_param_grad[c], pidx,
                    vec_get(mu_kld_param_grad[c], pidx) -
                    resp[c] * vec_get(prec_resid[c], pidx));
      }

      /* Explicit covariance derivative of log q_mix plus the pathwise
         covariance derivative through the selected component sample. */
      if (data->type == DIAG) {
        for (int pidx = 0; pidx < fulld; pidx++) {
          double selected_centered = vec_get(points, pidx) -
            mmvn_get_mu_el(mixmvn->components[component], pidx);
          for (int c = 0; c < mixmvn->ncomponents; c++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(mixmvn->components[c], pidx);
            double explicit_sigma = 0.5 * resp[c] *
              (1.0 - centered * vec_get(prec_resid[c], pidx));
            vec_set(sigma_kld_param_grad[c], pidx,
                    vec_get(sigma_kld_param_grad[c], pidx) + explicit_sigma);
          }
          vec_set(sigma_kld_param_grad[component], pidx,
                  vec_get(sigma_kld_param_grad[component], pidx) +
                  0.5 * vec_get(kld_dL_dx, pidx) * selected_centered);
        }
      }
      else {
        double loglambda_grad = 0.0;
        for (int c = 0; c < mixmvn->ncomponents; c++) {
          double quad = 0.0;
          for (int pidx = 0; pidx < fulld; pidx++) {
            double centered = vec_get(points, pidx) -
              mmvn_get_mu_el(mixmvn->components[c], pidx);
            quad += centered * vec_get(prec_resid[c], pidx);
          }
          vec_set(sigma_kld_param_grad[c], 0,
                  vec_get(sigma_kld_param_grad[c], 0) +
                  0.5 * resp[c] * (fulld - quad));
        }
        for (int pidx = 0; pidx < fulld; pidx++) {
          double selected_centered = vec_get(points, pidx) -
            mmvn_get_mu_el(mixmvn->components[component], pidx);
          loglambda_grad += 0.5 * vec_get(kld_dL_dx, pidx) *
            selected_centered;
        }
        vec_set(sigma_kld_param_grad[component], 0,
                vec_get(sigma_kld_param_grad[component], 0) +
                loglambda_grad);
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

        if (mu_kld_param_grad != NULL)
          vec_set(mu_kld_param_grad[component], pidx,
                  vec_get(mu_kld_param_grad[component], pidx) -
                  (fplus - fminus) / (2.0 * DERIV_EPS));
      }
      if (mu_kld_param_grad != NULL) {
        for (int c = 0; c < mixmvn->ncomponents; c++) {
          multi_MVN *comp = mixmvn->components[c];
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
            vec_set(mu_kld_param_grad[c], pidx,
                    vec_get(mu_kld_param_grad[c], pidx) -
                    (fplus - fminus) / (2.0 * DERIV_EPS));
          }
        }
      }
      for (int c = 0; c < mixmvn->ncomponents; c++) {
        for (int pidx = 0; pidx < data->covar_params[c]->size; pidx++) {
          double orig_param = vec_get(data->covar_params[c], pidx);
          double fplus, fminus;

          vec_set(data->covar_params[c], pidx, orig_param + DERIV_EPS);
          mixmvn_update_covariance(mixmvn, data);
          if (c == component) {
            lowr_map_std(mixmvn->components[component], points_std, points_tweak);
            fplus = mixmvn_log_dens(mixmvn, points_tweak);
            if (data->treeprior == NULL)
              fplus += 0.5 * (fulld * log(2 * M_PI) +
                              vec_inner_prod(points_tweak, points_tweak));
          }
          else {
            fplus = mixmvn_log_dens(mixmvn, points);
            if (data->treeprior == NULL)
              fplus += 0.5 * (fulld * log(2 * M_PI) +
                              vec_inner_prod(points, points));
          }

          vec_set(data->covar_params[c], pidx, orig_param - DERIV_EPS);
          mixmvn_update_covariance(mixmvn, data);
          if (c == component) {
            lowr_map_std(mixmvn->components[component], points_std, points_tweak);
            fminus = mixmvn_log_dens(mixmvn, points_tweak);
            if (data->treeprior == NULL)
              fminus += 0.5 * (fulld * log(2 * M_PI) +
                               vec_inner_prod(points_tweak, points_tweak));
          }
          else {
            fminus = mixmvn_log_dens(mixmvn, points);
            if (data->treeprior == NULL)
              fminus += 0.5 * (fulld * log(2 * M_PI) +
                               vec_inner_prod(points, points));
          }

          vec_set(data->covar_params[c], pidx, orig_param);
          mixmvn_update_covariance(mixmvn, data);
          vec_set(sigma_kld_param_grad[c], pidx,
                  vec_get(sigma_kld_param_grad[c], pidx) -
                  (fplus - fminus) / (2.0 * DERIV_EPS));
        }
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
   likelihood calculation in elbo_montecarlo while still handling
   E_q[log q_mix(x)], whose mixture entropy has no simple closed form. */
static double elbo_mix_kld_montecarlo(mixture_MVN *mixmvn, int component,
                                         CovarData *data, int nminibatch,
                                         double *kld,
                                         Vector **sigma_kld_param_grad,
                                         Vector **mu_kld_param_grad,
                                         Vector *weight_kld_param_grad) {
  int fulld = data->nseqs * data->dim;
  Vector *points = vec_new(fulld), *points_std;
  double kld_sum = 0.0;
  double scale = data->kld_upweight /
    (nminibatch * data->pointscale * data->pointscale);

  if (data->type == LOWR)
    points_std = vec_new(data->lowrank * data->dim);
  else
    points_std = vec_new(fulld);

  if (sigma_kld_param_grad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_zero(sigma_kld_param_grad[c]);
  if (mu_kld_param_grad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_zero(mu_kld_param_grad[c]);
  if (weight_kld_param_grad != NULL)
    vec_zero(weight_kld_param_grad);

  if (mixmvn->ncomponents == 1) {
    *kld = compute_closed_form_kld(mixmvn, data, fulld,
                                      sigma_kld_param_grad != NULL ? sigma_kld_param_grad[0] : NULL,
                                      mu_kld_param_grad != NULL ? mu_kld_param_grad[0] : NULL);
    vec_free(points);
    vec_free(points_std);
    return *kld;
  }

  for (int i = 0; i < nminibatch; i++) {
    sample_points(mixmvn->components[component], points, points_std, i);
    kld_sum += mix_kld_sample_grad(mixmvn, component, data, points,
                                      points_std, sigma_kld_param_grad,
                                      mu_kld_param_grad, weight_kld_param_grad);
  }

  *kld = kld_sum * scale;
  if (sigma_kld_param_grad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_scale(sigma_kld_param_grad[c], scale);
  if (mu_kld_param_grad != NULL)
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_scale(mu_kld_param_grad[c], scale);
  if (weight_kld_param_grad != NULL)
    vec_scale(weight_kld_param_grad, scale);

  vec_free(points);
  vec_free(points_std);
  return *kld;
}

double elbo_montecarlo(TreeModel *mod, mixture_MVN *mixmvn, int component,
                          CovarData *data,
                          int nminibatch, Vector *model_param_grad, Vector *nuis_param_grad,
                          double *ave_lprior, double *avemigll, double *kld,
                          Vector **sigma_kld_param_grad, Vector **mu_kld_param_grad,
                          Vector *weight_kld_param_grad) {
  Vector *sample_model_param_grad = vec_new(model_param_grad->size),
    *sample_nuis_param_grad = NULL, *points, *points_std;
  double ll, migll = 0, lprior = 0, avell = 0,
    ave_lprior_sum = 0, avemigll_sum = 0;
  int n = data->nseqs, dim = data->dim, fulld = n*dim;
  int estimate_kld = (kld != NULL);
  int estimate_kld_grad = (estimate_kld && sigma_kld_param_grad != NULL);
  double scale = data->kld_upweight /
    (nminibatch * data->pointscale * data->pointscale);

  vec_zero(model_param_grad);
  if (estimate_kld_grad) {
    for (int c = 0; c < mixmvn->ncomponents; c++)
      vec_zero(sigma_kld_param_grad[c]);
    if (mu_kld_param_grad != NULL)
      for (int c = 0; c < mixmvn->ncomponents; c++)
        vec_zero(mu_kld_param_grad[c]);
  }
  if (weight_kld_param_grad != NULL)
    vec_zero(weight_kld_param_grad);
  if (nuis_param_grad != NULL) {
    sample_nuis_param_grad = vec_new(nuis_param_grad->size);
    vec_zero(nuis_param_grad);
  }

  if (estimate_kld)
    *kld = mixmvn->ncomponents == 1 ?
      compute_closed_form_kld(mixmvn, data, fulld,
                                estimate_kld_grad ? sigma_kld_param_grad[0] : NULL,
                                mu_kld_param_grad != NULL ? mu_kld_param_grad[0] : NULL) : 0;

  points = vec_new(fulld);
  if (data->type == LOWR) /* in this case, the underlying standard
                             normal MVN is of the lower dimension */
    points_std = vec_new(data->lowrank * dim);
  else
    points_std = vec_new(fulld);

  for (int i = 0; i < nminibatch; i++) {
    migll = 0;
    lprior = 0;
    vec_zero(sample_model_param_grad);

    sample_points(mixmvn->components[component], points, points_std, i);
    /* Prior contribution to sample_model_param_grad is routed inside compute_model_grad
       via dL_dt -> Jacobian -> dL_dx; the prior's nuisance grads
       (relclock_sig_grad, nodetimes_grad) are then picked up below by
       update_nuis_grad. */
    ll = compute_model_grad(mod, mixmvn->components[component], points, points_std,
                               sample_model_param_grad, data,
                               component, NULL, &migll, &lprior);
    assert(isfinite(ll));

    avell += ll;
    avemigll_sum += migll;
    ave_lprior_sum += lprior;
    if (estimate_kld && mixmvn->ncomponents > 1)
      *kld += mix_kld_sample_grad(mixmvn, component, data, points,
                                     points_std,
                                     estimate_kld_grad ? sigma_kld_param_grad : NULL,
                                     estimate_kld_grad ? mu_kld_param_grad : NULL,
                                     weight_kld_param_grad);
    vec_plus_eq(model_param_grad, sample_model_param_grad);

    if (nuis_param_grad != NULL) {
      vec_zero(sample_nuis_param_grad);
      update_nuis_grad(mod, data, sample_nuis_param_grad);
      vec_plus_eq(nuis_param_grad, sample_nuis_param_grad);
    }
  }

  /* divide by nminibatch to get expected gradient */
  vec_scale(model_param_grad, 1.0/nminibatch);
  avell /= nminibatch;
  ave_lprior_sum /= nminibatch;
  avemigll_sum /= nminibatch;
  if (ave_lprior != NULL)
    *ave_lprior = ave_lprior_sum;
  if (avemigll != NULL)
    *avemigll = avemigll_sum;
  if (estimate_kld && mixmvn->ncomponents > 1)
    *kld *= scale;
  if (estimate_kld_grad) {
    if (mixmvn->ncomponents > 1) {
      for (int c = 0; c < mixmvn->ncomponents; c++)
        vec_scale(sigma_kld_param_grad[c], scale);
      if (mu_kld_param_grad != NULL)
        for (int c = 0; c < mixmvn->ncomponents; c++)
          vec_scale(mu_kld_param_grad[c], scale);
    }
  }
  if (weight_kld_param_grad != NULL)
    vec_scale(weight_kld_param_grad, scale);

  /* same for nuisance grad if needed */
  if (nuis_param_grad != NULL)
    vec_scale(nuis_param_grad, 1.0 / nminibatch);

  /* free everything and return */
  vec_free(points); vec_free(points_std); vec_free(sample_model_param_grad);
  if (nuis_param_grad != NULL)
    vec_free(sample_nuis_param_grad);

  /* we also have to free the last tree in the tree model to avoid a
     memory leak */
  tr_free(mod->tree);
  mod->tree = NULL;
  
  return avell;
}

/* Wrapper for various distance-based tree inference algorithms. */
TreeNode *infer_distance_tree(Matrix *D, char **names, Matrix *dt_dD, Neighbors *nb,
                              CovarData *data) {
  if (data->ultrametric) {
    TreeNode *t = upgma_infer_tree(D, names, dt_dD);

    if (data->no_zero_br == TRUE)
      repair_zero_br(t);
    return t;
  }
  else {
    TreeNode *tree = nj_infer_tree(D, names, dt_dD, nb);
    if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
      if (data->seq_to_node_map == NULL)
        update_seq_to_node_map(tree, names, data);
      if (data->tree_diam_leaf1 < 0 || data->tree_diam_leaf2 < 0)
        update_diam_leaves(D, data);  /* needed upon init */
      TreeNode *mp = tr_find_midpoint(tree, data->seq_to_node_map[data->tree_diam_leaf1],
                                      data->seq_to_node_map[data->tree_diam_leaf2]);
      TreeNode *newtree = tr_reroot2(tree, mp);
      tree = newtree;
    }
    return tree;
  }
}

/* sample a list of trees from the approximate posterior distribution
   and return as a new list.  If logdens is non-null, return
   corresponding vector of log densities for the samples */
List *var_sample(int nsamples, mixture_MVN *mixmvn, CovarData *data, char** names,
                    Vector *logdens) {
  List *retval = lst_new_ptr(nsamples);
  int i, component;
  TreeNode *tree;
  Vector *points_x = vec_new(data->dim * data->nseqs),
    *points_y = vec_new(data->dim * data->nseqs);
  
  for (i = 0; i < nsamples; i++) {
    component = mixmvn_sample_component(mixmvn);
    sample_points(mixmvn->components[component], points_x, NULL, 0);
    
    if (logdens != NULL) 
      vec_set(logdens, i, mixmvn_log_dens(mixmvn, points_x));
     
    apply_normalizing_flows(points_y, points_x, data, component, NULL);
    points_to_distances(points_y, data);
    tree = infer_distance_tree(data->dist, names, NULL, NULL, data);
    lst_push_ptr(retval, tree);
  }
  
  vec_free(points_x);
  vec_free(points_y);
  return(retval);
}

/* return a single tree representing the approximate posterior mean */
TreeNode *mean_tree(Vector *mu, char **names, CovarData *data) {
  TreeNode *tree;
  
  if (data->nseqs * data->dim != mu->size)
    die("ERROR in mean_tree: bad dimensions\n");

  points_to_distances(mu, data);  
  tree = infer_distance_tree(data->dist, names, NULL, NULL, data);
  
  return(tree);
}

/* sample points from variational distribution.  This is a wrapper
   that encapsulates the use of antithetic sampling.  If points_std is
   non-NULL, it will be used to store the baseline standard normal
   variate for use in downstream calculations in variational
   inference. Antithetic sampling is only used in this case */
void sample_points(multi_MVN *mmvn, Vector *points, Vector *points_std,
                      int sample_idx) {
  static Vector *cachedpoints = NULL, *cachedstd = NULL;  

  if (points_std == NULL) 
    mmvn_sample(mmvn, points); /* simple in this case */
  else {
    /* otherwise we have to make use of caching for antithetic sampling */
    if (cachedpoints != NULL &&
        (cachedpoints->size != points->size ||
         cachedstd->size != points_std->size)) {
      vec_free(cachedpoints);
      vec_free(cachedstd);
      cachedpoints = NULL; /* force realloc */
    }
    if (cachedpoints == NULL) {
      cachedpoints = vec_new(points->size);
      cachedstd = vec_new(points_std->size);
    }
    
    if (sample_idx % 2 == 0) { /* new sample, update caches */
      mmvn_sample_anti_keep(mmvn, points, cachedpoints, points_std);
      vec_copy(cachedstd, points_std);

    }
    else { /* just use cache to define sample */
      vec_copy(points, cachedpoints);
      vec_copy(points_std, cachedstd);
      vec_scale(points_std, -1.0);
    }
  }
}

/* given points_x, apply normalizing flows to compute points_y as y =
   f(x).  Optionally populates *logdet with total log determinate of
   Jacobian (if non-NULL) */
void apply_normalizing_flows(Vector *points_y, Vector *points_x,
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
void set_kld_sigma_grad_LOWR(Vector *sigma_kld_param_grad, multi_MVN *mmvn) {
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
      vec_set(sigma_kld_param_grad, i*mmvn->mvn->lowR->ncols + j,
              mat_get(Rgrad, i, j));

  mat_free(Rgrad);
}

/* compute partial derivatives of entropy H[q(x)] wrt LOWR variance
   parameters: Sigma_0 = I + R R^T, Sigma = I_d ⊗ Sigma_0. */
void set_entropy_sigma_grad_LOWR(Vector *sigma_entgrad, multi_MVN *mmvn) {
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

/* optimize variational model by stochastic gradient ascent using the
   Adam algorithm.  Takes initial tree model and alignment and
   distance matrix, dimensionality of Euclidean space to work in.
   Note: alters distance matrix */
void variational_inf(TreeModel *mod, mixture_MVN *mixmvn, int nminibatch,
                        double learnrate, int nbatches_conv, int min_nbatches,
                        CovarData *data, FILE *logf,
                        unsigned int silent, unsigned int log_all) {

  Vector *model_param_grad, **var_component_param_grad = NULL,
    **sigma_penalty_param_grad = NULL,
    *weight_param_grad = NULL, *weight_kld_param_grad = NULL,
    *m_weight = NULL, *v_weight = NULL, *best_logits = NULL,
    *component_elbo = NULL, *tmp_model_param_grad = NULL, *tmp_nuis_param_grad = NULL,
    *tmp_weight_kld_param_grad = NULL, **tmp_mu_kld_param_grad = NULL, **tmp_sigma_kld_param_grad = NULL, 
    **m_mu = NULL, **v_mu = NULL, **best_mu = NULL,
    **m_sigma = NULL, **v_sigma = NULL, *gauge_reference_mu = NULL,
    **best_sigmapar = NULL;
  int n = data->nseqs, j, k, t = 0, stop = FALSE, bestt = -1, graddim,
    dim = data->dim, fulld = n*dim, reenable_taylor_t = -1,
    ncomponents = mixmvn->ncomponents, weight_t = 0, mig_warmup_active = 0;
  int *sigma_t = NULL;
  double elb = 0, avell, avemigll, kld, bestelb = -INFTY, bestll = -INFTY,
    bestkld = -INFTY, bestmigll = -INFTY,
    running_tot = 0, last_running_tot = -INFTY, penalty = 0,
    bestpenalty = 0, ave_lprior, best_lprior = -INFTY, subsamp_rescale = 1.0,
    ll_at_mean = 0, bestll_at_mean = -INFTY, grad_norm_sq;
  TaylorData *taylor_stash = NULL;

  /* for nuisance parameters; these are parameters that are optimized
     by stochastic gradient descent but are not fully sampled via the
     variational distribution */
  int n_nuisance_params = get_num_nuisance_params(mod, data);
  Vector *nuis_param_grad = NULL, *m_nuis = NULL, *v_nuis = NULL,
    *m_nuis_prev = NULL, *v_nuis_prev = NULL, *best_nuis_params = NULL;
  if (mixmvn->components[0]->d * mixmvn->components[0]->n != dim * n)
    die("ERROR in variational_inf: bad dimensions\n");

  graddim = fulld + data->covar_params[0]->size;
  model_param_grad = vec_new(graddim);
  var_component_param_grad = smalloc(ncomponents * sizeof(Vector*));
  weight_param_grad = vec_new_zero(ncomponents);
  weight_kld_param_grad = vec_new_zero(ncomponents);
  m_weight = vec_new_zero(ncomponents);
  v_weight = vec_new_zero(ncomponents);
  best_logits = vec_new(ncomponents);
  component_elbo = vec_new(ncomponents);
  tmp_model_param_grad = vec_new(graddim);
  tmp_weight_kld_param_grad = vec_new(ncomponents);
  tmp_mu_kld_param_grad = smalloc(ncomponents * sizeof(Vector*));
  tmp_sigma_kld_param_grad = smalloc(ncomponents * sizeof(Vector*));
  vec_copy(best_logits, mixmvn->logits);
  sigma_t = smalloc(ncomponents * sizeof(int));
  m_mu = smalloc(ncomponents * sizeof(Vector*));
  v_mu = smalloc(ncomponents * sizeof(Vector*));
  m_sigma = smalloc(ncomponents * sizeof(Vector*));
  v_sigma = smalloc(ncomponents * sizeof(Vector*));
  best_mu = smalloc(ncomponents * sizeof(Vector*));
  best_sigmapar = smalloc(ncomponents * sizeof(Vector*));
  sigma_penalty_param_grad = smalloc(ncomponents * sizeof(Vector*));

  if (n_nuisance_params > 0) {
    nuis_param_grad = vec_new(n_nuisance_params);
    m_nuis = vec_new_zero(n_nuisance_params);
    v_nuis = vec_new_zero(n_nuisance_params);
    m_nuis_prev = vec_new_zero(n_nuisance_params);
    v_nuis_prev = vec_new_zero(n_nuisance_params);
    best_nuis_params = vec_new(n_nuisance_params);
    tmp_nuis_param_grad = vec_new(n_nuisance_params);
  }

  /* initialize component-specific moments and iteration counts*/
  for (k = 0; k < ncomponents; k++) {
    var_component_param_grad[k] = vec_new_zero(graddim);
    m_mu[k] = vec_new_zero(fulld);
    v_mu[k] = vec_new_zero(fulld);
    m_sigma[k] = vec_new_zero(data->covar_params[k]->size);
    v_sigma[k] = vec_new_zero(data->covar_params[k]->size);
    best_mu[k] = vec_new(fulld);
    best_sigmapar[k] = vec_new(data->covar_params[k]->size);
    sigma_penalty_param_grad[k] = vec_new_zero(graddim);
    tmp_mu_kld_param_grad[k] = vec_new_zero(fulld);
    tmp_sigma_kld_param_grad[k] = vec_new_zero(data->covar_params[k]->size);
    mmvn_save_mu(mixmvn->components[k], best_mu[k]);
    vec_copy(best_sigmapar[k], data->covar_params[k]);
    sigma_t[k] = 0;
  }

  gauge_reference_mu = gauge_fixing_new_reference(mixmvn, data);
  gauge_fixing_apply(mixmvn, data, gauge_reference_mu, m_mu);
  for (k = 0; k < ncomponents; k++)
    mmvn_save_mu(mixmvn->components[k], best_mu[k]);

  log_header(mod, mixmvn, data, logf, fulld,
                         n_nuisance_params, log_all);

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

  /* note the configured migration warmup phase if needed */
  if (data->crispr_mod != NULL && data->migtable != NULL && !silent) {
    fprintf(stderr, "Running %d warmup iterations without migration "
            "model...\n", CPR_MIG_WARMUP_ITERS);
    data->crispr_mod->mig_warmup = TRUE;
    mig_warmup_active = TRUE;
  }

  do {
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

    /* zero gradients */
    vec_zero(weight_param_grad);
    vec_zero(weight_kld_param_grad);
    vec_zero(component_elbo);
    for (k = 0; k < mixmvn->ncomponents; k++) {
      vec_zero(var_component_param_grad[k]);
      vec_zero(sigma_penalty_param_grad[k]);
      assert(var_component_param_grad[k]->size == graddim);
    }
    if (nuis_param_grad != NULL)
      vec_zero(nuis_param_grad);
    
    /* configure subsampling */
    if (!sd->full_grad_now && data->crispr_mod == NULL && data->nthreads == 1) {
      data->subsample = TRUE;
      data->subsampsize = sd->m;
      data->reuse_subsamp = !sd->resample_sites;
      subsamp_rescale = (double)data->msa->length / data->subsampsize;
    }
    else {
      data->subsample = FALSE;
      subsamp_rescale = 1.0;
    }

    /* decide whether to re-enable Taylor approximation */
    if (taylor_stash != NULL && data->taylor == NULL &&
        t == reenable_taylor_t) {
      if (!silent)
        fprintf(stderr, "WARNING: re-enabling Taylor approximation.\n");
      data->taylor = taylor_stash;
      taylor_stash = NULL;
    }

    /* turn off migration warmup if needed */
    if (data->crispr_mod != NULL && data->migtable != NULL && 
        t == CPR_MIG_WARMUP_ITERS && !silent) {
      fprintf(stderr, "Warmup complete; enabling migration model...\n");
      data->crispr_mod->mig_warmup = FALSE;
      mig_warmup_active = FALSE;
    }

    /* per-component elbo computation */
    kld = 0.0;
    avell = ave_lprior = avemigll = ll_at_mean = elb = penalty = 0.0;
    double cavell, cave_lprior, cavemigll, cll_at_mean, celb, cpenalty, wk, ckld;
    int c;
    for (k = 0; k < ncomponents; k++) {
      cavell = 0.0;
      cave_lprior = 0.0;
      cavemigll = 0.0;
      cll_at_mean = 0.0;
      wk = vec_get(mixmvn->weights, k);
      ckld = 0.0;

      vec_zero(tmp_model_param_grad);
      if (tmp_nuis_param_grad != NULL)
        vec_zero(tmp_nuis_param_grad);

      if (data->taylor != NULL) {
        cavell = elbo_hybrid(mod, mixmvn, k, data, nminibatch,
                              tmp_model_param_grad, tmp_nuis_param_grad,
                              &cave_lprior, &cavemigll, &cll_at_mean);
        if ((data->crispr_mod != NULL &&
            data->crispr_mod->zero_likl == TRUE) || !isfinite(cavell)) {
          if (!silent)
            fprintf(stderr, "WARNING: Taylor approximation produced invalid "
                    "likelihood; switching to Monte Carlo.\n");
          reenable_taylor_t = t + 10;
          taylor_stash = data->taylor;
          data->taylor = NULL;
        }
        else {
          elbo_mix_kld_montecarlo(mixmvn, k, data, nminibatch,
                                    &ckld, tmp_sigma_kld_param_grad,
                                    tmp_mu_kld_param_grad, tmp_weight_kld_param_grad);
        }
      }

      if (data->taylor == NULL) {
        cll_at_mean = 0.0;
        cavell = elbo_montecarlo(mod, mixmvn, k, data, nminibatch,
                                  tmp_model_param_grad, tmp_nuis_param_grad,
                                  &cave_lprior, &cavemigll, &ckld,
                                  tmp_sigma_kld_param_grad, tmp_mu_kld_param_grad,
                                  tmp_weight_kld_param_grad);
      }

      /* re-scale if subsampling */
      if (data->subsample == TRUE) {
        vec_scale(tmp_model_param_grad, subsamp_rescale);
        if (tmp_nuis_param_grad != NULL)
          vec_scale(tmp_nuis_param_grad, subsamp_rescale);
        cavell *= subsamp_rescale;
      }

      /* compute component variance penalty */
      cpenalty = compute_variance_penalty(sigma_penalty_param_grad[k],
                                                    mixmvn->components[k],
                                                    data, k);

      /* compute the total elbo for this component */
      celb = cavell + cave_lprior - ckld - cpenalty + cavemigll;
      vec_set(component_elbo, k, celb);

      /* accumulate mixture weighted elbo components */
      avell += wk * cavell;
      ave_lprior += wk * cave_lprior;
      kld += wk * ckld;
      penalty += wk * cpenalty;
      avemigll += wk * cavemigll;
      ll_at_mean += wk * cll_at_mean;
      elb += wk * celb;

      for (j = 0; j < graddim; j++)
        vec_set(var_component_param_grad[k], j,
                vec_get(var_component_param_grad[k], j) +
                wk * vec_get(tmp_model_param_grad, j) +
                wk * vec_get(sigma_penalty_param_grad[k], j));

      /* fold this component's weighted contribution to the mixture-KLD
         gradient directly into every component's parameter gradient; log
         q_mix depends on every component density, so a sample drawn from
         component k contributes to all c's gradients. */
      for (c = 0; c < ncomponents; c++) {
        for (j = 0; j < fulld; j++)
          vec_set(var_component_param_grad[c], j,
                  vec_get(var_component_param_grad[c], j) +
                  wk * vec_get(tmp_mu_kld_param_grad[c], j));
        for (j = 0; j < data->covar_params[c]->size; j++)
          vec_set(var_component_param_grad[c], fulld + j,
                  vec_get(var_component_param_grad[c], fulld + j) +
                  wk * vec_get(tmp_sigma_kld_param_grad[c], j));
      }

      /* gradient for weight parameters */
      if (ncomponents > 1) {
        for (j = 0; j < ncomponents; j++)
          vec_set(weight_kld_param_grad, j, vec_get(weight_kld_param_grad, j) +
                  wk * vec_get(tmp_weight_kld_param_grad, j));
      }

      if (nuis_param_grad != NULL)
        for (j = 0; j < n_nuisance_params; j++)
          vec_set(nuis_param_grad, j, vec_get(nuis_param_grad, j) +
                  wk * vec_get(tmp_nuis_param_grad, j));
    }

    if (ncomponents > 1) {
      elb = add_mixture_weight_dirichlet_prior(mixmvn, component_elbo,
                                                weight_kld_param_grad, weight_param_grad,
                                                elb);
    }

    /* don't select best during migration warmup: migration is excluded from
       the ELBO then, making warmup ELBOs artificially high and incomparable
       to post-warmup ELBOs that include the migration log likelihood */
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
      for (k = 0; k < ncomponents; k++) {
        mmvn_save_mu(mixmvn->components[k], best_mu[k]);
        vec_copy(best_sigmapar[k], data->covar_params[k]);
      }
      vec_copy(best_logits, mixmvn->logits);
      if (n_nuisance_params > 0)
        save_nuis_params(best_nuis_params, mod, data);
    }

    /* Rescale the complete component gradient in place by approximate inverse
       Fisher information to put parameters on similar optimization scales. */
    for (k = 0; k < ncomponents; k++) {
      if (data->natural_grad == TRUE)
        rescale_grad(var_component_param_grad[k],
                        var_component_param_grad[k], mixmvn->components[k], data);
    }
    /* we won't do this with nuisance params */

    /* update scheduler with norm of the actual Adam update and clip if
       necessary. */
    grad_norm_sq = 0.0;
    for (k = 0; k < ncomponents; k++) {
      for (j = 0; j < fulld; j++) {
        double g = vec_get(var_component_param_grad[k], j);
        grad_norm_sq += g * g;
      }
      for (j = 0; j < data->covar_params[k]->size; j++) {
        double g = vec_get(var_component_param_grad[k], fulld + j);
        grad_norm_sq += g * g;
      }
    }
    if (ncomponents > 1)
      for (j = 0; j < ncomponents; j++) {
        double g = vec_get(weight_param_grad, j);
        grad_norm_sq += g * g;
      }
    sm->grad_norm = sqrt(grad_norm_sq);
    if (sd->clip_norm > 0 && sm->grad_norm > sd->clip_norm) {
      clip_scale = sd->clip_norm / sm->grad_norm;
      for (k = 0; k < ncomponents; k++)
        vec_scale(var_component_param_grad[k], clip_scale);
      clipped = TRUE;
    }

    /* Adam updates; see Kingma & Ba, arxiv 2014 */
    t++;
    data->variational_iter = t; /* useful for debugging in other routines */

    /* Update mu.  In mixture mode every component receives its weighted
       likelihood/prior gradient; all components also receive mixture-KLD mean
       gradients because log q_mix depends on every component density. */
    for (k = 0; k < ncomponents; k++) {
      for (j = 0; j < fulld; j++) {
        double m = vec_get(m_mu[k], j), v = vec_get(v_mu[k], j);
        double g = vec_get(var_component_param_grad[k], j);
        mmvn_set_mu_el(mixmvn->components[k], j,
                        adam_scalar_update(mmvn_get_mu_el(mixmvn->components[k], j),
                                           &m, &v, t, g, sd->lr));
        vec_set(m_mu[k], j, m);
        vec_set(v_mu[k], j, v);
      }
    }

    /* Fix the gauge symmetry between mixture components.
        This could be done in the single component case too, but
        it would add runtime which we want to avoid for now.*/
    if (ncomponents > 1) {
      gauge_fixing_apply(mixmvn, data, gauge_reference_mu, m_mu);
    }

    /* Update sigma.  In mixture mode every component receives its weighted
       likelihood/prior gradient; all components also receive mixture-KLD
       covariance gradients because log q_mix depends on every component
       density. */
    for (k = 0; k < ncomponents; k++) {
      sigma_t[k]++;
      for (j = 0; j < data->covar_params[k]->size; j++) {
        double m = vec_get(m_sigma[k], j), v = vec_get(v_sigma[k], j);
        double g = vec_get(var_component_param_grad[k], fulld + j);
        vec_set(data->covar_params[k], j,
                adam_scalar_update(vec_get(data->covar_params[k], j),
                                   &m, &v, sigma_t[k], g, sd->lr));
        vec_set(m_sigma[k], j, m);
        vec_set(v_sigma[k], j, v);
      }
    }
    mixmvn_update_covariance(mixmvn, data);

    if (ncomponents > 1) {
      weight_t++;
      for (j = 0; j < ncomponents; j++) {
        double m = vec_get(m_weight, j), v = vec_get(v_weight, j);
        double g = clip_scale * vec_get(weight_param_grad, j);
        vec_set(mixmvn->logits, j,
                adam_scalar_update(vec_get(mixmvn->logits, j),
                                   &m, &v, weight_t, g, sd->lr));
        vec_set(m_weight, j, m);
        vec_set(v_weight, j, v);
      }
      mixmvn_update_weights(mixmvn);
    }

    /* same thing for nuisance params, if necessary */
    for (j = 0; j < n_nuisance_params; j++) {   
      double g, m, v, old_val, new_val;

      if (!nuis_param_is_active(mod, data, j, ncomponents))
        continue;

      g = vec_get(nuis_param_grad, j);
      m = vec_get(m_nuis_prev, j);
      v = vec_get(v_nuis_prev, j);
      old_val = nuis_param_get(mod, data, j);
      new_val = adam_scalar_update(old_val, &m, &v, t, g,
                                   sd->lr * 0.3);
      nuis_param_pluseq(mod, data, j, new_val - old_val);
      vec_set(m_nuis, j, m);
      vec_set(v_nuis, j, v);
      /* factor of 0.3 above to slow learning of nuisance params */
    }
    if (n_nuisance_params > 0) {
      vec_copy(m_nuis_prev, m_nuis);
      vec_copy(v_nuis_prev, v_nuis);
    }
    
    log_iteration(mod, mixmvn, data, logf, taylor_stash, sm,
                              t, avell, elb, ave_lprior, kld,
                              penalty, avemigll, clipped,
                              n_nuisance_params, log_all);
    
    /* check total elb every nbatches_conv to decide whether to stop */
    running_tot += elb;
    if (t % nbatches_conv == 0) {
      log_running_elbo(logf, nbatches_conv, running_tot);
      if ((sd->full_grad_now || data->crispr_mod != NULL) && t >= min_nbatches &&
          1.001*running_tot <= last_running_tot*0.999)
        /* sometimes get stuck increasingly asymptotically; stop if increase not more than about 0.1% */
        stop = TRUE;

      last_running_tot = running_tot;
      running_tot = 0;
    }    
  } while(stop == FALSE);

  /* Revert to the best parameters found after convergence */
  for (k = 0; k < ncomponents; k++) {
    mmvn_set_mu(mixmvn->components[k], best_mu[k]);
    vec_copy(data->covar_params[k], best_sigmapar[k]);
  }
  vec_copy(mixmvn->logits, best_logits);
  mixmvn_update_weights(mixmvn);
  mixmvn_update_covariance(mixmvn, data);
  if (n_nuisance_params > 0)
    update_nuis_params(best_nuis_params, mod, data);

  /* if using Taylor approximation, run one final MC pass at the restored
     best parameters to get an unbiased estimate of E[lnL] for reporting.
     The hybrid ELBO used during training can be biased (especially when
     variance is at floor), so this gives an accurate final value. */
  double final_mc_ll = 0;
  if (data->taylor != NULL && logf != NULL) {
    for (k = 0; k < ncomponents; k++) {
      double component_mc_ll =
        elbo_montecarlo(mod, mixmvn, k, data, nminibatch,
                           model_param_grad, nuis_param_grad,
                           NULL, NULL, NULL, NULL, NULL, NULL);
      final_mc_ll += vec_get(mixmvn->weights, k) * component_mc_ll;
    }
  }

  log_final(mod, mixmvn, data, logf, bestt, bestelb, bestll,
                        best_lprior, bestkld, bestpenalty, bestll_at_mean,
                        final_mc_ll, bestmigll, n_nuisance_params);

  if (!silent) fprintf(stderr, "Converged in %d iterations; ELBO=%.2f...\n", t, bestelb);

  vec_free(model_param_grad);
  vec_free(weight_param_grad);
  vec_free(weight_kld_param_grad);
  vec_free(m_weight);
  vec_free(v_weight);
  vec_free(best_logits);
  vec_free(component_elbo);
  vec_free(tmp_model_param_grad);
  vec_free(tmp_weight_kld_param_grad);
  if (gauge_reference_mu != NULL)
    vec_free(gauge_reference_mu);
  for (k = 0; k < ncomponents; k++) {
    vec_free(var_component_param_grad[k]);
    vec_free(m_mu[k]);
    vec_free(v_mu[k]);
    vec_free(m_sigma[k]);
    vec_free(v_sigma[k]);
    vec_free(best_mu[k]);
    vec_free(best_sigmapar[k]);
    vec_free(sigma_penalty_param_grad[k]);
    vec_free(tmp_mu_kld_param_grad[k]);
    vec_free(tmp_sigma_kld_param_grad[k]);
  }
  sfree(var_component_param_grad);
  sfree(m_mu);
  sfree(v_mu);
  sfree(m_sigma);
  sfree(v_sigma);
  sfree(best_mu);
  sfree(best_sigmapar);
  sfree(sigma_penalty_param_grad);
  sfree(tmp_mu_kld_param_grad);
  sfree(tmp_sigma_kld_param_grad);
  sfree(sigma_t);
  sfree(s); sfree(st); sfree(sd); sfree(sm);
  
  if (n_nuisance_params > 0) {
    vec_free(nuis_param_grad); vec_free(m_nuis); vec_free(v_nuis);
    vec_free(m_nuis_prev); vec_free(v_nuis_prev); vec_free(best_nuis_params);
    vec_free(tmp_nuis_param_grad);
  }    
}
