/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* calculation of gradients for SGA */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>

/* compute the gradient of the log likelihood for a tree model with
   respect to the free parameters of the MVN averaging distribution,
   starting from a given MVN sample (points). Returns log likelihood
   of current model, which is computed as a by-product.  */
double nj_compute_model_grad(TreeModel *mod, multi_MVN *mmvn, Vector *points,
                             Vector *points_std, Vector *grad, CovarData *data,
                             double *nf_logdet,
                             double *migll) {
  int n = data->nseqs; /* number of taxa */
  int d = mmvn->n * mmvn->d / n; /* dimensionality; have to accommodate diagonal case */
  int dim = n*d; /* full dimension of point vector */
  int i, j, k;
  double porig, ll_base, loglambda_grad;
  Vector *dL_dx = vec_new(dim);
  
  if (grad->size != dim + data->params->size)
    die("ERROR in nj_compute_model_grad: bad gradient dimension.\n");
  
  if (data->type == DIST && data->Lapl_pinv_evals == NULL)
    die("ERROR in nj_compute_model_grad: Laplacian pseudoinverse and eigendecomposition required in DIST case.\n");
  else if (data->type == LOWR && data->R == NULL)
    die("ERROR in nj_compute_model_grad: low-rank matrix R required in LOWR case.\n");
  
  /* obtain gradient with respect to points, dL/dx */
  ll_base = nj_dL_dx_smartest(points, dL_dx, mod, data, nf_logdet, migll);

  /* TEMPORARY: compare with numerical version */
  /* printf("analytical (%f):\n", ll_base); */
  /* vec_print(dL_dx, stdout); */
  /* ll_base = nj_dL_dx_dumb(points, dL_dx, mod, data); */
  /* printf("numerical (%f):\n", ll_base); */
  /* vec_print(dL_dx, stdout); */
  /* exit(0); */

  if (!isfinite(ll_base)) /* can happen with crispr model; force calling code to deal with it */
    return ll_base;
        
  /* now derive partial derivatives wrt free parameters from dL/dx */
  vec_zero(grad);
  loglambda_grad = 0;
  for (i = 0; i < n; i++) {  
    for (k = 0; k < d; k++) {
      int pidx = i*d + k;
      porig = vec_get(points, pidx);

      /* the partial derivative wrt the mean parameter is equal to the
         derivative with respect to the point, because the point is just a
         translation of a 0-mean MVN variable via the reparameterization
         trick */
      vec_set(grad, pidx, vec_get(dL_dx, pidx));

      /* the partial derivative wrt the variance parameter is more
         complicated because of the reparameterization trick */      
      if (data->type == CONST || data->type == DIST)
        loglambda_grad += 0.5 * vec_get(dL_dx, pidx) * (porig - mmvn_get_mu_el(mmvn, pidx));
      /* assumes log parameterization of scale factor lambda */
      
      else if (data->type == DIAG) 
        /* in the DIAG case, the partial derivative wrt the
           corresponding variance parameter can be computed directly
           based on a single point and coordinate */
        vec_set(grad, (i+n)*d + k, 0.5 * vec_get(dL_dx, pidx) * (porig - mmvn_get_mu_el(mmvn, pidx))); 
      
    }
  }
  if (data->type == CONST || data->type == DIST)  /* in this case, need to update the final
                                                      gradient component corresponding to the
                                                      lambda parameter */
    vec_set(grad, dim, loglambda_grad);

  else if (data->type == LOWR) { /* in this case have to sum across
                                    dimensions because there is a
                                    many-to-many relationship with the
                                    variance parameters */
    for (i = 0; i < n; i++) 
      for (j = 0; j < data->lowrank; j++) 
        for (k = 0; k < d; k++) 
          vec_set(grad, dim + i*data->lowrank + j, vec_get(grad, dim + i*data->lowrank + j) +
                  vec_get(grad, i*d + k) * vec_get(points_std, j*d + k));
          /* update for parameter corresponding to element R[i, j],
             which has index in grad of dim + (i*data->lowrank) + j.
             The gradient is a dot product of dL/dx and dx/dp, where L
             is the log likelihood, x is the point, and p is the
             variance parameter.  In this case, dxj/dp is simply zj, the
             corresponding standardized variable.
          */
  }

  vec_free(dL_dx);
  return ll_base;
}  

/* version of nj_compute_model_grad that calculates all derivatives
   numerically, to check correctness of analytical calculations.  The
   check knows nothing about the details of the parameterization; it
   just uses a brute force numerical calculation */
double nj_compute_model_grad_check(TreeModel *mod, multi_MVN *mmvn, 
                                   Vector *points, Vector *points_std,
                                   Vector *grad, CovarData *data) {
  int n = data->msa->nseqs; /* number of taxa */
  int d = mmvn->n * mmvn->d / n; /* dimensionality; have to accommodate diagonal case */
  int dim = n*d; /* full dimension of point vector */
  int i, j;
  double porig, ll_base, ll, deriv;
  TreeNode *tree, *orig_tree;   /* has to be rebuilt repeatedly; restore at end */
  Vector *points_tweak = vec_new(points->size);
  Vector *sigmapar = data->params;
  Vector *dL_dx = vec_new(points->size);
  Matrix *D = data->dist;
  
  if (grad->size != dim + data->params->size)
    die("ERROR in nj_compute_model_grad_check: bad gradient dimension.\n");
  
  /* set up tree model and get baseline log likelihood */
  nj_points_to_distances(points, data);    
  tree = nj_inf(D, data->names, NULL, data);
  orig_tree = tr_create_copy(tree);   /* restore at the end */
  nj_reset_tree_model(mod, tree);
  ll_base = nj_compute_log_likelihood(mod, data, NULL);

  if (!isfinite(ll_base)) /* can happen with crispr model; force
                             calling code to deal with it */
    return ll_base;
  
  /* Now perturb each point and propagate perturbation through distance
     calculation, neighbor-joining reconstruction, and likelihood
     calculation on tree */
  for (i = 0; i < dim; i++) {
    double mu_orig, dxi_dmui;

    porig = vec_get(points, i);
    vec_set(points, i, porig + DERIV_EPS);

    nj_points_to_distances(points, data); 
    tree = nj_inf(D, data->names, NULL, data);
    nj_reset_tree_model(mod, tree);      
    ll = nj_compute_log_likelihood(mod, data, NULL);

    if (!isfinite(ll)) /* can happen with crispr model; force
                          calling code to deal with it */
      return ll;

    deriv = (ll - ll_base) / DERIV_EPS; 
    vec_set(dL_dx, i, deriv); /* derive of log likelihood wrt dim i of
                                 point; need to save this for later */
    
    /* the mean is straightforward but check it anyway to be sure */
    mu_orig = mmvn_get_mu_el(mmvn, i);
    mmvn_set_mu_el(mmvn, i, mu_orig + DERIV_EPS);
    vec_copy(points_tweak, points_std);
    mmvn_map_std(mmvn, points_tweak);
    dxi_dmui = (vec_get(points_tweak, i) - porig) / DERIV_EPS;
    /* should be 1! */
    vec_set(grad, i, deriv * dxi_dmui);
    mmvn_set_mu_el(mmvn, i, mu_orig);
            
    vec_set(points, i, porig); /* restore orig */
  }

  /* for the variance parameters, we have to consider potential
     changes to entire vector of points */
  for (i = 0; i < sigmapar->size; i++) {
    double dL_dp = 0, origp = vec_get(sigmapar, i);    
    vec_set(sigmapar, i, origp + DERIV_EPS);
    nj_update_covariance(mmvn, data);
    vec_copy(points_tweak, points_std);
    mmvn_map_std(mmvn, points_tweak);
    vec_minus_eq(points_tweak, points);
    vec_scale(points_tweak, 1.0/DERIV_EPS); /* now contains dx / dp
                                               where p is the variance
                                               parameter */

    /* dL/dp is a dot product of dL/dx and dx/dp */
    for (j = 0; j < points_tweak->size; j++)
      dL_dp += vec_get(dL_dx, j) * vec_get(points_tweak, j);
    
    vec_set(grad, i + dim, dL_dp);
    vec_set(sigmapar, i, origp); /* restore orig */
  }
  
  nj_update_covariance(mmvn, data); /* make sure to leave it in original state */

  nj_reset_tree_model(mod, orig_tree);
  vec_free(points_tweak);
  vec_free(dL_dx);
  return ll_base;
}  

/* rescale gradients by approximate inverse Fisher information for approx
   natural gradient scale */
void nj_rescale_grad(Vector *grad, Vector *rsgrad, multi_MVN *mmvn, CovarData *data) {
  int i, j, fulld = mmvn->n * mmvn->d;
  for (i = 0; i < grad->size; i++) {
    double g = vec_get(grad, i);
    
    if (i < fulld) { /* mean gradients */
      if (data->type == CONST || data->type == DIAG) 
        g *= mat_get(mmvn->mvn->sigma, i, i); /* these are all the same in the CONST case */
      else { /* DIST or LOWR */ /* CHECK: this code untested */
        /* scale by dot product of corresponding row of sigma with the original gradient */
        double dotp = 0.0;
        int sigmarow = i / mmvn->d;  /* project down to sigma */
        int d = i % mmvn->d; /* corresponding dimension */
        for (j = 0; j < mmvn->mvn->sigma->ncols; j++)
          dotp += mat_get(mmvn->mvn->sigma, sigmarow, j) * vec_get(grad, j*mmvn->d + d);
        g *= dotp;
      }
    }
    else { /* variance gradients */
      if (data->type == CONST)
        g *= 2/(mmvn->n * mmvn->d); /* assumes variance is exp(parameter) */
      else if (data->type == DIAG)
        g *= 2; /* assumes variance is exp(parameter) */
      else if (data->type == DIST)
        g *= 2/(mmvn->n-1); /* assumes variance is exp(parameter) */
      else
        break; /* handle LOWR case below */
    }
    
    vec_set(rsgrad, i, g);
  }

  if (data->type == LOWR) { /* CHECK: this code as yet untested */
    /* in this case the rescaled variance gradients can be obtained by
       matrix multiplication with sigma */

    /* first coerce the relevant gradient components into an n x k matrix */
    Matrix *Rgrad = mat_new(mmvn->n, data->lowrank),
      *rsRgrad = mat_new(mmvn->n, data->lowrank);
    for (i = 0; i < mmvn->n; i++)
      for (j = 0; j < data->lowrank; j++)
        mat_set(Rgrad, i, j, vec_get(grad, fulld + i*data->lowrank + j));

    /* multiply on the left by sigma */
    mat_mult(rsRgrad, mmvn->mvn->sigma, Rgrad);

    /* finally extract the rescaled values */
    for (i = 0; i < mmvn->n; i++)
      for (j = 0; j < data->lowrank; j++)
        vec_set(rsgrad, fulld + i*data->lowrank + j, mat_get(rsRgrad, i, j));
    
    mat_free(Rgrad);
    mat_free(rsRgrad);
  }
}

/* Compute penalty for variance and its gradient  */
void nj_compute_variance_penalty(Vector *grad, multi_MVN *mmvn,
                               CovarData *data) {  
  int i, j, n = mmvn->n, d = mmvn->d;
  int start_idx = n*d;  /* starting index for variance parameters */
  
  if (data->type == CONST || data->type == DIST) {
    /* L2 penalty applied to log lambda (which is the free parameter) */
    double loglambda = log(data->lambda);
    double mult = data->var_reg * PENALTY_LOGLAMBDA_CONST;  /* constant applied to log lambda */
    data->var_pen = mult * loglambda * loglambda; /* L2 penalty */
    vec_set(grad, start_idx, -2.0 * mult * loglambda); /* derivative of L2 penalty; 
                                                          negative because penalty 
                                                          will be subtracted */
  }
  else if (data->type == DIAG) {
    /* similar to above but L2 penalty applied to each diagonal element */
    data->var_pen = 0;
    double mult = data->var_reg * PENALTY_LOGLAMBDA_DIAG;
    for (i = 0; i < data->params->size; i++) {
      double loglambda_i = vec_get(data->params, i);
      data->var_pen += mult * loglambda_i * loglambda_i;
      vec_set(grad, start_idx + i, -2.0 * mult * loglambda_i); 
    }
  }
  else {
    assert (data->type == LOWR);

    /* in this case, the analog is to use the sum of squared log
       eigenvalues of the embedded low-rank matrix */
    MVN *Rmvn = mmvn->mvn->lowRmvn;
    assert(Rmvn->evals != NULL);
    data->var_pen = 0;
    double mult = data->var_reg * PENALTY_LOGLAMBDA_LOWR;
    for (i = 0; i < Rmvn->dim; i++) {
      double logeval = log(vec_get(Rmvn->evals, i));
      data->var_pen += mult * logeval * logeval;
    }
    
    /* the gradient for this penalty is 2*mult * R * V * diag(log
       eval/eval) * V^T, where R is the low-rank matrix and V comes
       from the eigendecomposition of R R^T. */
    Matrix *Rgrad = mat_new(data->R->nrows, data->R->ncols);
    Matrix *tmp = mat_new(Rmvn->dim, Rmvn->dim);
    Vector *logdiag = vec_new(Rmvn->dim);
    for (i = 0; i < Rmvn->dim; i++) {
      double eval = vec_get(Rmvn->evals, i);
      if (eval < 1.0e-6) eval = 1.0e-6;
      vec_set(logdiag, i, log(eval) / eval);
    }
    mat_mult_diag_transp(tmp, Rmvn->evecs, logdiag);
    mat_mult(Rgrad, mmvn->mvn->lowR, tmp);
    mat_scale(Rgrad, 2 * mult);
    vec_free(logdiag);   
    
    /* finally add entries to corresponding gradient components */
    vec_zero(grad);
    for (i = 0; i < data->R->nrows; i++)
      for (j = 0; j < data->R->ncols; j++)
        vec_set(grad, start_idx + i*data->R->ncols + j,
                vec_get(grad, start_idx + i*data->R->ncols + j) - 
                mat_get(Rgrad, i, j));
    /* note have to subtract because makes negative contribution */
    
    mat_free(Rgrad);
    mat_free(tmp);
  }
}

/* alternative versions of gradient calculation. Can be cross-checked
   for debugging */

/* compute the gradient of the log likelihood with respect to the
   individual points by a very simple, fully numerical method.
   Returns log likelihood of model as by product. */
double nj_dL_dx_dumb(Vector *x, Vector *dL_dx, TreeModel *mod, 
                     CovarData *data) {
  double ll, ll_base, xorig, deriv;
  int i, k;
  int n = data->nseqs; /* number of taxa */
  int d = data->dim; /* dimensionality; have to accommodate diagonal case */
  TreeNode *tree, *orig_tree;   /* has to be rebuilt repeatedly; restore at end */

  assert(data->msa != NULL && data->crispr_mod == NULL);
  /* this version not set up for crispr data */
         
  /* set up tree model and get baseline log likelihood */
  nj_points_to_distances(x, data);    
  tree = nj_inf(data->dist, data->msa->names, NULL, data);
  orig_tree = tr_create_copy(tree);   /* restore at the end */
  nj_reset_tree_model(mod, tree);
  ll_base = nj_compute_log_likelihood(mod, data, NULL);

  /* Now perturb each point and propagate perturbation through distance
     calculation, neighbor-joining reconstruction, and likelihood
     calculation on tree */  
  for (i = 0; i < n; i++) {  
    for (k = 0; k < d; k++) {
      int idx = i*d + k;

      xorig = vec_get(x, idx);
      vec_set(x, idx, xorig + DERIV_EPS);

      nj_points_to_distances(x, data); 
      tree = nj_inf(data->dist, data->msa->names, NULL, data);
      nj_reset_tree_model(mod, tree);      
      ll = nj_compute_log_likelihood(mod, data, NULL);
      
      deriv = (ll - ll_base) / DERIV_EPS; 

      vec_set(dL_dx, idx, deriv);
      vec_set(x, idx, xorig); /* restore orig */
    }
  }
  nj_reset_tree_model(mod, orig_tree);
  return ll_base;
}

/* compute the gradient of the log likelihood with respect to the
   individual branch lengths.  This version uses numerical methods
   (mostly useful for testing analytical version) */
double nj_dL_dt_num(Vector *dL_dt, TreeModel *mod, CovarData *data) {
  int nodeidx;
  double ll, ll_base;
  List *traversal;

  if (data->crispr_mod != NULL)
    ll_base = cpr_compute_log_likelihood(data->crispr_mod, NULL);
  else
    ll_base = nj_compute_log_likelihood(mod, data, NULL);

  if (!isfinite(ll_base)) /* can happen with crispr; force calling
                             code to deal with it */
    return ll_base;
  
  /* perturb each branch and recompute likelihood */
  traversal = mod->tree->nodes;
  assert(dL_dt->size == lst_size(traversal) - 1); 
  vec_zero(dL_dt);
  for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
    TreeNode *node = lst_get_ptr(traversal, nodeidx);
    double orig_t = node->dparent;

    if (node == mod->tree || node == mod->tree->rchild)
      continue; /* only consider one branch beneath the root because
                   implicitly unrooted */
    
    node->dparent += DERIV_EPS;

    if (data->crispr_mod != NULL)
      ll = cpr_compute_log_likelihood(data->crispr_mod, NULL);
    else
      ll = nj_compute_log_likelihood(mod, data, NULL);

    if (!isfinite(ll)) /* can happen with crispr; force calling
                          code to deal with it */
      return ll;
    
    vec_set(dL_dt, nodeidx, (ll - ll_base) / DERIV_EPS);
    node->dparent = orig_t;
  }
  return ll_base;  
}

/* compute the Jacobian matrix for 2n-3 branch lengths wrt n-choose-2
   pairwise distances.  This version uses numerical methods and is
   intended for validation of the analytical version */
void nj_dt_dD_num(Matrix *dt_dD, Matrix *D, TreeModel *mod, CovarData *data) {
  TreeNode *tree, *orign, *node;
  int i, j, n = data->msa->nseqs, nodeidx;
  List *trav_tree, *trav_orig;
  
  /* perturb each pairwise distance and measure effect on each branch */
  trav_orig = mod->tree->nodes;
  mat_zero(dt_dD);
  for (i = 0; i < n; i++) {
    for (j = i+1; j < n; j++) {
      double orig_d = mat_get(D, i, j);
      mat_set(D, i, j, orig_d + DERIV_EPS);
      tree = nj_inf(D, data->msa->names, NULL, data);

      /* compare the trees, branch by branch */
      /* we will assume the same topology although that will
         occasionally not be true; good enough for sanity checking */
      trav_tree = tree->nodes;
      for (nodeidx = 0; nodeidx < lst_size(trav_orig); nodeidx++) {
        node = lst_get_ptr(trav_tree, nodeidx);

        if (node == tree || node == tree->rchild) /* unrooted tree */
          continue;
        
        orign = lst_get_ptr(trav_orig, nodeidx);
        
        if (node->id != orign->id) continue;
        
        mat_set(dt_dD, nodeidx, nj_i_j_to_dist(i, j, n),
                (node->dparent - orign->dparent) / DERIV_EPS);
      }
      
      mat_set(D, i, j, orig_d);
      tr_free(tree);
    }
  }
}

/* compute the gradient of the log likelihood with respect to the
   individual points by the chain rule and using analytical methods
   for each component.  Fastest but most complicated and error-prone
   version. */
double nj_dL_dx_smartest(Vector *x, Vector *dL_dx, TreeModel *mod,
                         CovarData *data, double *nj_logdet, double *migll) {
  int n = data->nseqs, nbranches = 2*n-2,  /* have to work with the rooted tree here */
    ndist = n * (n-1) / 2, ndim = data->nseqs * data->dim;
  Vector *dL_dt = vec_new(nbranches);
  Matrix *dt_dD = mat_new(nbranches, ndist);
  Vector *dL_dD = vec_new(ndist);
  Vector *dL_dy = vec_new(dL_dx->size);
  Vector *migbranchgrad = data->migtable != NULL ?
    vec_new(nbranches) : NULL;
  Vector *y = vec_new(x->size);
  TreeNode *tree;
  double ll_base;
  int i, j, d;

  *migll = 0.0;
  
  /* convert x to y using normalizing flows if available */
  nj_apply_normalizing_flows(y, x, data, nj_logdet);
  
   /* set up baseline objects */
  nj_points_to_distances(y, data);
  tree = nj_inf(data->dist, data->names, dt_dD, data);
  nj_reset_tree_model(mod, tree);

  /* TEMPORARY: compare to numerical gradient */
  /* fprintf(stdout, "dt_dD (analytical):\n"); */
  /* mat_print(dt_dD, stdout); */
  /* nj_dt_dD_num(dt_dD, data->dist, mod, data); */
  /* fprintf(stdout, "dt_dD (numerical):\n"); */
  /* mat_print(dt_dD, stdout); */
  /* exit(0); */

  /* calculate log likelihood and analytical gradient */
  if (data->crispr_mod != NULL)
    ll_base = cpr_compute_log_likelihood(data->crispr_mod, dL_dt);
  else
    ll_base = nj_compute_log_likelihood(mod, data, dL_dt);

  if (!isfinite(ll_base)) /* can happen with crispr; force calling
                             code to deal with it */
    return ll_base;
  
  /* TEMPORARY: compare dL_dt to numerical version */
  /* fprintf(stdout, "dL_dt (analytical):\n"); */
  /* vec_print(dL_dt, stdout); */
  /* nj_dL_dt_num(dL_dt, mod, data); */
  /* fprintf(stdout, "dL_dt (numerical):\n"); */
  /* vec_print(dL_dt, stdout); */
  /* exit(0); */

  /* also get migration log likelihood if needed */
  if (data->migtable != NULL) {
    *migll = mig_compute_log_likelihood(mod, data->migtable, data->crispr_mod,
                                        migbranchgrad);
    vec_plus_eq(dL_dt, migbranchgrad);
  }

  /* apply chain rule to get dL/dD gradient (a vector of dim ndist) */
  mat_vec_mult_transp(dL_dD, dt_dD, dL_dt);
  /* (note taking transpose of both vector and matrix and expressing
     result as column vector) */

  /* finally multiply by dD/dy to obtain gradient wrt y.  This part is
     different for the euclidean and hyperbolic geometries */
  vec_zero(dL_dy);

  if (data->hyperbolic) {
    
    /* first precompute x0[i] = sqrt(1 + ||x_i||^2) */
    double *x0 = (double*)smalloc(n * sizeof(double));
    double alpha = 1.0 / sqrt(data->negcurvature);   /* curvature radius */
    for (i = 0; i < n; i++) {
      double ss = 1.0;
      int base = i * data->dim;
      for (d = 0; d < data->dim; d++) {
        double xid = vec_get(x, base + d);
        ss += xid * xid;
      }
      x0[i] = sqrt(ss);
    }

    /* accumulate pairwise contributions */
    for (i = 0; i < n; i++) {
      double denom_inv, pref;
      int base_i = i * data->dim;
      for (j = i + 1; j < n; j++) {
        int base_j = j * data->dim;

        /* weight = dL/dD_ij */
        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));

        /* down-weight saturated pairs with large distance */
        double Dij = mat_get(data->dist, i, j);
        if (Dij > 10) weight *= (10 / Dij);           /* soft clip: in (0,1] */
 
        /* u = x0_i*x0_j - <x_i, x_j>  (equals -Lorentz inner product) */
        double dot_spatial = 0.0;
        for (d = 0; d < data->dim; d++) 
          dot_spatial += vec_get(x, base_i + d) * vec_get(x, base_j + d);

        /* prefactor; clamp sqrt(u^2 - 1) for stability */
        double u = x0[i] * x0[j] - dot_spatial;
        
        denom_inv = d_acosh_du_stable(u);         /* = d/du acosh(u) */        
        pref = (alpha / data->pointscale) * denom_inv;
        
        /* dD/dx_i and dD/dx_j contributions */
        for (d = 0; d < data->dim; d++) {
          int idx_i = base_i + d;
          int idx_j = base_j + d;
          double xid = vec_get(x, idx_i);
          double xjd = vec_get(x, idx_j);

          double gi = pref * (-xjd + (x0[j] / x0[i]) * xid);  /* dD_ij/dx_i^d */
          double gj = pref * (-xid + (x0[i] / x0[j]) * xjd);  /* dD_ij/dx_j^d */

          /* accumulate weighted by w_ij = dL/dD_ij */
          vec_set(dL_dy, idx_i, vec_get(dL_dy, idx_i) + weight * gi);
          vec_set(dL_dy, idx_j, vec_get(dL_dy, idx_j) + weight * gj);
        }
      }
    }

    /* add a small radius prior to prevent points from "ballooning" away from zero */
    const double lambda_base = 1e-5; 
    const double lambda_eff  = lambda_base / (data->pointscale*data->pointscale);
      
    for (i = 0; i < ndim; i++)
      vec_set(dL_dy, i, vec_get(dL_dy, i) - 2.0 * lambda_eff * vec_get(x, i));

    sfree(x0);

    /* in this case normalizing flows are not allowed, so we'll just copy directly into dL/dx */
    vec_copy(dL_dx, dL_dy);
  }
  else { /* euclidean version is simpler */
    for (i = 0; i < n; i++) {
      int base_i = i * data->dim;
      for (j = i + 1; j < n; j++) {
        int base_j = j * data->dim;
        double dist_ij = mat_get(data->dist, i, j);
        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));

        if (dist_ij < 1e-15) dist_ij = 1e-15;
        
        for (d = 0; d < data->dim; d++) {
          int idx_i = base_i + d;
          int idx_j = base_j + d;
            
          double coord_diff = vec_get(x, idx_i) - vec_get(x, idx_j);
          double grad_contrib = weight * coord_diff / (dist_ij * data->pointscale * data->pointscale);
          /* need two factors of pointscale, one for the coord_diff, one for the distance */
          
          vec_set(dL_dy, idx_i, vec_get(dL_dy, idx_i) + grad_contrib);
          vec_set(dL_dy, idx_j, vec_get(dL_dy, idx_j) - grad_contrib);
        }
      }
    }

    /* in case of normalizing flows, need one more step in the chain rule.
       Note that this is supported only in the Euclidean case for now;
       need to move this call outside of 'else' if hyperbolic support
       is added */
    if (data->rf != NULL && data->pf != NULL) {
      /* apply both; need temporary vector */
      Vector *tmp = vec_new(dL_dx->size);
      rf_backprop(data->rf, x, tmp, dL_dy);
      pf_backprop(data->pf, x, dL_dx, tmp);
      vec_free(tmp);
    }    
    else if (data->rf != NULL) 
      /* We need to back-propagate through the radial flow to obtain
         the real dL_dx */
      rf_backprop(data->rf, x, dL_dx, dL_dy);
      /* note that the gradients wrt the parameters a, b, and ctr are
         computed as side-effects and stored inside rf */
    else if (data->pf != NULL) 
      pf_backprop(data->pf, x, dL_dx, dL_dy);
      /* similar for planar flow */
    else
      vec_copy(dL_dx, dL_dy);
  }

  vec_free(dL_dt);
  mat_free(dt_dD);
  vec_free(dL_dD);
  vec_free(dL_dy);
  vec_free(y);
  if (migbranchgrad != NULL) vec_free(migbranchgrad);

  return ll_base;  
}

