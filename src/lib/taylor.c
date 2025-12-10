/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* ELBO estimation based on Taylor approximation to reduce number of
   NJ calls */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <taylor.h>
#include <upgma.h>
#include <gradients.h>
#include <likelihoods.h>
#include <hutchinson.h>
#include <nuisance.h>
#include <backprop.h>
#include <geometry.h>

TaylorData *tay_new(CovarData *data) {
  TaylorData *td = smalloc(sizeof(TaylorData));
  td->covar_data = data;
  td->nbranches = data->nseqs * 2 - 2; /* CHECK */
  td->nx = data->nseqs * data->dim;    /* necessary? */

  /* allocate workspace memory, CHECK */
  td->base_grad = vec_new(td->nbranches);
  td->Jbx = mat_new(td->nbranches, td->nx);
  td->JbxT = mat_new(td->nx, td->nbranches);
  td->tmp_x1 = vec_new(td->nx);
  td->tmp_x2 = vec_new(td->nbranches);
  
  return td;
}

void tay_free(TaylorData *td) {
  vec_free(td->base_grad);
  mat_free(td->Jbx);
  mat_free(td->JbxT);
  vec_free(td->tmp_x1);
  vec_free(td->tmp_x2);
  sfree(td);
}

/* estimate key components of the ELBO by a Taylor approximation
   around the mean.  Returns the expected log likelihood */
double nj_elbo_taylor(TreeModel *mod, multi_MVN *mmvn, CovarData *data,
                      Vector *grad, Vector *nuis_grad, double *lprior,
                      double *migll) {
  double ll;
  Vector *prior_grad = NULL;

  if (data->treeprior != NULL) 
    prior_grad = vec_new(grad->size);  
  
  /* first calculate log likelihood at the mean */
  vec_zero(grad);
  Vector *mu = vec_new(mmvn->n * mmvn->d), *mu_std = vec_new(mmvn->n * mmvn->d);
  mmvn_save_mu(mmvn, mu); /* express mean as a single vector */
  mmvn_rederive_std(mmvn, mu, mu_std); /* rederive associated standard normal variate */
  /* CHECK: not sure mu_std is really needed here */

  /* FIXME: do I have to apply the flows here?  or does nj_compute_model_grad
     do that internally? */

  /* FIXME: I need a tree model representing the mean.  Whill that be set
     up correctly inside nj_compute_model_grad? */
  
  ll = nj_compute_model_grad(mod, mmvn, mu, mu_std, grad, data, NULL, migll);
  /* CHECK: is handling of mean correct in this case?  several terms
     reduce to zero but maybe ok.  Maybe variance is wrong here
     however? */

  /* also calculate log prior if needed */
  if (data->treeprior != NULL) {
    (*lprior) = tp_compute_log_prior(mod, data, prior_grad);
    vec_plus_eq(grad, prior_grad);
  }

  /* build Jbx and JbxT once at the mean point */
  tay_prep_jacobians(data->taylor, mod, mu);
 
  if (nuis_grad != NULL) {
    vec_zero(nuis_grad);
    nj_update_nuis_grad(mod, data, nuis_grad);
  }


  
  /* note that there is no first-order term in the Taylor approximation
     because we are expanding around the mean */
  
  /* CHECK: do we need to propagate gradients wrt to the variance
     terms through to the migll?  how about nuisance params? */
  /* CHECK: are there also second order terms to consider for the log prior? */

  /* now add the second-order terms for the Taylor expansion.  These
     terms are equal to 1/2 tr(H Sigma), where H is the Hessian of the
     ELBO.  But we can simplify this expression by considering the
     chain of transformations from the standard normal to the
     phylogeny and likelihood.  The NJ transformation is linear up to
     a choice of neighbors, the tranformation from z to x is linear.
     If we also assume the distances are locally linear, then all
     curvature comes from the phylogenetic likelihood function, and we
     can approximate tr(H Sigma) by tr(H S), where S is a square
     matrix of dimension nbranches x nbranches representing a product
     of Sigma and the relevant Jacobian matrices [see
     manuscript for detailed derivation] */
 
  /* we will approximate tr(H S) using a Hutchinson trace estimator */
  double tr = hutch_tr(tay_HVP, tay_SVP, data->taylor, grad->size, NHUTCH_SAMPLES);
  /* FIXME: check dimensions */

  /* FIXME: apply constant of 1/2.  should we return this value or add it to ll? */
  
  /* CHECK: should we consider the curvature of the flows? */
     
  /* free everything and return */
  vec_free(mu); vec_free(mu_std);
  if (data->treeprior != NULL)
    vec_free(prior_grad);
 
  return ll; 
}

/* helper functions for Hessian-vector product computation */

/* save current branch lengths from tree into vector bl */
static inline
void tr_save_branch_lengths(TreeNode *root, Vector *bl) {
  assert(root->nnodes == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    vec_set(bl, i, n->dparent);
  }
}

/* adjust branch lengths by incrementing scaled values in vector bl;
   excludes root */
static inline
void tr_incr_branch_lengths(TreeNode *root, Vector *bl, double scale) {
  assert(root->nnodes == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    if (n->parent == NULL)
      continue; /* skip root */
    n->dparent += (scale * vec_get(bl, i));
    if (n->dparent < 1e-6)
      n->dparent = 1e-6; /* prohibit zero or negative lengths */
  }
}

/* restore branch lengths from vector bl */
static inline
void tr_restore_branch_lengths(TreeNode *root, Vector *bl) {
  assert(root->nnodes == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    n->dparent = vec_get(bl, i);
  }
}

/* Finite-difference version of Hessian-vector product based on
   directional gradient.  Computes H v ≈ (grad(theta + eps*v) -
   grad(theta)) / eps */
void tay_HVP(Vector *out, Vector *v, void *dat) {

  TaylorData *tay_data = (TaylorData *)dat;
  CovarData *data = tay_data->covar_data;
  TreeModel *mod  = data->mod;  /* FIXME: have to populate */

  Vector *origbl = vec_new(mod->tree->nnodes);
  tr_save_branch_lengths(mod->tree, origbl);
  /* all branch lengths including root are stored, but root is not
     modified */

  tr_incr_branch_lengths(mod->tree, v, DERIV_EPS);
  /* excludes root, which is ignored in gradient calculations */
  /* CHECK: does v have correct dimension? need to skip root or look up by id? */

  if (data->crispr_mod != NULL) 
    cpr_compute_log_likelihood(data->crispr_mod, out);
  else 
    nj_compute_log_likelihood(mod, data, out);

  vec_minus_eq(out, tay_data->base_grad); /* FIXME: have to populate base_grad */
  vec_scale(out, 1.0 / DERIV_EPS);

  tr_restore_branch_lengths(mod->tree, origbl);

  vec_free(origbl);
}

/* Compute S_b * v, where S_b = Jbx * Sigma_x * Jbx^T.
   v is a vector in branch-length space (dimension = nbranches).
   out is also in branch-length space.
   data_vd is CovarData*, which must contain Jbx, Sigma, and workspace vectors. */
void tay_SVP(Vector *out, Vector *v, void *dat) {
  TaylorData *tay_data = (TaylorData *)dat;
  CovarData *data = tay_data->covar_data;
  int n = data->nseqs, nbranches = 2*n-2, ndim = data->nseqs * data->dim;
    
  assert(v->size == nbranches);
  assert(out->size == nbranches);

  /* tmp_x1 = JbxT * v */
  mat_vec_mult(tay_data->tmp_x1, tay_data->JbxT, v);

  /* tmp_x2 = Sigma * tmp_x_1 */
  tay_sigma_vec_mult(tay_data->tmp_x2, tay_data->mmvn, tay_data->tmp_x1);

  /* out = Jbx * tmp_x2 */
  mat_vec_mult(out, tay_data->Jbx, tay_data->tmp_x2);
}

/* FIXME: move this to gradients.c ? */
void tay_prep_jacobians(TaylorData *tay_data, TreeModel *mod, Vector *x_mean) {
  int nb = tay_data->nbranches;
  int nx = tay_data->nx;

  if (tay_data->Jbx == NULL)
    tay_data->Jbx  = mat_new(nb, nx);

  if (tay_data->JbxT == NULL)
    tay_data->JbxT = mat_new(nx, nb);

  /* Workspace for reverse-mode J^T e_j */
  Vector *dL_dt = vec_new(nb);
  Vector *dL_dx = vec_new(nx);
  /* CHECK: use CovarData workspaces instead? */

  /* Loop over each branch length index */
  for (int j = 0; j < nb; j++) {

    /* dL_dt = e_j */
    vec_zero(dL_dt);
    vec_set(dL_dt, j, 1.0);

    /* Use existing reverse-mode path: 
       dL_dt → dL_dD → dL_dy → dL_dx 
       evaluated at the mean point.
    */
    tay_dx_from_dt(dL_dt, dL_dx, mod, tay_data);

    /* Fill column j of Jbx^T */
    for (int k = 0; k < nx; k++)
      mat_set(tay_data->JbxT, k, j, vec_get(dL_dx, k));
  }

  /* Jbx = transpose(JbxT) */
  mat_trans(tay_data->Jbx, tay_data->JbxT);

  vec_free(dL_dt);
  vec_free(dL_dx);
}

/* Compute J_{bx}^T * dL_dt at the MEAN point.
   Inputs:
   dL_dt : (nbranches)    vector in branch-length space (seed direction)
   Output:
   dL_dx : (n*d)           vector in latent coordinate space
   Uses:
   data->nb      neighbor tape (from mean tree)
   data->dist    distances at mean
   data->y       embedding at mean
   data->rf, pf  flows
   Everything else must be precomputed.

   This is JUST the reverse-mode Jacobian chain for the mean point.
*/
void tay_dx_from_dt(Vector *dL_dt, Vector *dL_dx, TreeModel *mod,
                    TaylorData *tay_data) {
  CovarData *data = tay_data->covar_data;
  int n     = data->nseqs;
  int dim   = data->dim;
  int ndist = n*(n-1)/2;
  int nx    = tay_data->nx;

  /* Workspace from CovarData */
  Vector *dL_dD = tay_data->tmp_dD;   /* size ndist */
  Vector *dL_dy = tay_data->tmp_dy;   /* size nx    */

  vec_zero(dL_dD);
  vec_zero(dL_dy);
  vec_zero(dL_dx);

  /*--------------------------------------
   * 1. Branch lengths → distances (reverse)
   *--------------------------------------*/

  if (tay_data->nb != NULL)
    nj_dL_dD_from_neighbors(tay_data->nb, dL_dt, dL_dD);
  else
    upgma_dL_dD_from_tree(mod->tree, dL_dt, dL_dD);

  /*--------------------------------------
   * 2. Distances → embedding y  (reverse)
   *--------------------------------------*/

  if (data->hyperbolic) {
    /* hyperbolic reverse-mode */
    int i, j, d;

    /* Precompute needed geometric constants */
    double *x0 = (double *)smalloc(n * sizeof(double));
    Vector *y  = tay_data->y;         /* size nx */
    Matrix *dist = data->dist;    /* n x n  */

    for (i = 0; i < n; i++) {
      double ss = 1.0;
      int base = i * dim;
      for (d = 0; d < dim; d++) {
        double yi = vec_get(y, base + d);
        ss += yi * yi;
      }
      x0[i] = sqrt(ss);
    }

    for (i = 0; i < n; i++) {
      int base_i = i * dim;

      for (j = i+1; j < n; j++) {
        int base_j = j * dim;

        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));
        if (weight == 0.0) continue;

        double Dij = mat_get(dist, i, j);
        if (Dij > 10) weight *= (10 / Dij);

        /* Compute u and derivatives as in smarter() */
        double dot_spatial = 0.0;
        for (d = 0; d < dim; d++)
          dot_spatial += vec_get(y, base_i + d) *
            vec_get(y, base_j + d);

        double u = x0[i] * x0[j] - dot_spatial;
        double denom_inv = d_acosh_du_stable(u);
        double pref = (1.0 / sqrt(data->negcurvature))
          * (denom_inv / data->pointscale);

        for (d = 0; d < dim; d++) {
          int idx_i = base_i + d;
          int idx_j = base_j + d;

          double yi = vec_get(y, idx_i);
          double yj = vec_get(y, idx_j);

          double gi = pref * (-yj + (x0[j]/x0[i])*yi);
          double gj = pref * (-yi + (x0[i]/x0[j])*yj);

          vec_set(dL_dy, idx_i,
                  vec_get(dL_dy, idx_i) + weight * gi);
          vec_set(dL_dy, idx_j,
                  vec_get(dL_dy, idx_j) + weight * gj);
        }
      }
    }
    sfree(x0);

    /* no flows in hyperbolic case */
    vec_copy(dL_dx, dL_dy);
  }

  else {  /* Euclidean reverse-mode */

    int i, j, d;
    Vector *y = tay_data->y;
    Matrix *dist = data->dist;

    for (i = 0; i < n; i++) {
      int base_i = i * dim;

      for (j = i+1; j < n; j++) {
        int base_j = j * dim;

        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));
        if (weight == 0.0) continue;

        double Dij = mat_get(dist, i, j);
        if (Dij < 1e-15) Dij = 1e-15;

        for (d = 0; d < dim; d++) {

          int idx_i = base_i + d;
          int idx_j = base_j + d;

          double diff = vec_get(y, idx_i) - vec_get(y, idx_j);
          double grad_contrib =
            weight * diff /
            (Dij * data->pointscale * data->pointscale);

          vec_set(dL_dy, idx_i,
                  vec_get(dL_dy, idx_i) + grad_contrib);
          vec_set(dL_dy, idx_j,
                  vec_get(dL_dy, idx_j) - grad_contrib);
        }
      }
    }

    /* Backprop through flows: y → x */
    if (data->rf != NULL && data->pf != NULL) {
      Vector *tmp = tay_data->tmp_extra;
      rf_backprop(data->rf, y, tmp, dL_dy);
      pf_backprop(data->pf, y, dL_dx, tmp);
    }
    else if (data->rf != NULL) {
      rf_backprop(data->rf, y, dL_dx, dL_dy);
    }
    else if (data->pf != NULL) {
      pf_backprop(data->pf, y, dL_dx, dL_dy);
    }
    else {
      vec_copy(dL_dx, dL_dy);
    }
  }
}

/* Sigma * v for diagonal Sigma */
void tay_sigma_vec_mult(Vector *out, multi_MVN *mmvn, Vector *v)
{
  int nx = v->size;
  assert(out->size == nx);
  
  for (int i = 0; i < nx; i++) {
    double s = 1; /* exp(mmvn->eta[i]);*/  /* FIXME: make this parameterization general */
    vec_set(out, i, s * vec_get(v, i));
  }
}
