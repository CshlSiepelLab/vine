/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* various supporting functions for handling of covariance matrices
   and related data */
  
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <covariance.h>
#include <nj.h>
#include <geometry.h>
#include <phast/eigen.h>
#include <phast/markov_matrix.h>

/* update covariance matrix based on the parameters and auxiliary
   data */
void nj_update_covariance(multi_MVN *mmvn, CovarData *data) {
  int i, j;
  Vector *sigma_params = data->params;
  
  /* Note: variance parameters now stored as log values and must
     be exponentiated */
  mat_zero(mmvn->mvn->sigma);
  if (data->type == CONST) {
    mat_set_identity(mmvn->mvn->sigma);
    data->lambda = exp(vec_get(sigma_params, 0));
    if (!isfinite(data->lambda) || data->lambda < VARFLOOR) {
      data->lambda = VARFLOOR;
      vec_set(sigma_params, 0, log(VARFLOOR)); /* keeps param from running away */
    }
    mat_scale(mmvn->mvn->sigma, data->lambda);
  }
  else if (data->type == DIAG) {
    for (i = 0; i < sigma_params->size; i++) {
      double lambda_i = exp(vec_get(sigma_params, i));
      if (lambda_i < VARFLOOR) {
        lambda_i = VARFLOOR;
        vec_set(sigma_params, i, log(VARFLOOR));
      }
      mat_set(mmvn->mvn->sigma, i, i, lambda_i);
    }
  }
  else if (data->type == DIST) {
    data->lambda = VARFLOOR + exp(vec_get(sigma_params, 0));
    mat_copy(mmvn->mvn->sigma, data->Lapl_pinv);
    mat_scale(mmvn->mvn->sigma, data->lambda);

    if (mmvn->mvn->evecs == NULL) {
      mmvn->mvn->evecs = mat_new(mmvn->n, mmvn->n);
      mmvn->mvn->evals = vec_new(mmvn->n);
    }
    mat_copy(mmvn->mvn->evecs, data->Lapl_pinv_evecs); /* can simply derive eigendecomposition 
                                                          from Lapl_pinv */
    vec_copy(mmvn->mvn->evals, data->Lapl_pinv_evals);
    vec_scale(mmvn->mvn->evals, data->lambda);
  }
  else {
    assert(data->type == LOWR);
    for (i = 0; i < data->R->nrows; i++)
      for (j = 0; j < data->R->ncols; j++)
        mat_set(data->R, i, j, vec_get(data->params,
                                       i*data->R->ncols + j));
                                 /* note not log in this case */

    /* update the mvn accordingly */
    mvn_reset_LOWR(mmvn->mvn, data->R);
    /* it will do the Cholesky by default but we also need to force an
       eigendecomposition */
    mvn_preprocess(mmvn->mvn->lowRmvn, TRUE);    
  }
}

/* create a new CovarData object appropriate for the choice of
   parameterization */
CovarData *nj_new_covar_data(enum covar_type covar_param, Matrix *dist, int dim,
                             MSA *msa, CrisprMutModel* crispr_mod, char **names,
                             unsigned int natural_grad, double kld_upweight,
                             int rank, double var_reg, unsigned int hyperbolic,
                             double negcurvature, unsigned int ultrametric,
                             unsigned int radial_flow, unsigned int planar_flow,
                             TreePrior *treeprior, MigTable *mtable) {
  static int seeded = 0;
  
  CovarData *retval = smalloc(sizeof(CovarData));
  retval->type = covar_param;
  retval->msa = msa;
  retval->crispr_mod = crispr_mod;
  retval->names = names;
  retval->lambda = LAMBDA_INIT;
  retval->mvn_type = MVN_DIAG;
  retval->dist = dist;
  retval->nseqs = dist->nrows;
  retval->dim = dim;
  retval->natural_grad = natural_grad;
  retval->kld_upweight = kld_upweight;
  retval->Lapl_pinv = NULL;
  retval->Lapl_pinv_evals = NULL;
  retval->Lapl_pinv_evecs = NULL;
  retval->lowrank = -1;
  retval->R = NULL;
  retval->var_reg = var_reg;
  retval->hyperbolic = hyperbolic;
  retval->negcurvature = negcurvature;
  retval->ultrametric = ultrametric;
  retval->no_zero_br = FALSE;
  retval->treeprior = treeprior;
  retval->subsample = FALSE;
  retval->tree_diam_leaf1 = -1;
  retval->tree_diam_leaf2 = -1;
  retval->seq_to_node_map = NULL;
  retval->migtable = mtable;
  retval->gtr_params = NULL;
  retval->deriv_gtr = NULL;
  
  if (radial_flow == TRUE) {
    retval->rf = rf_new(retval->nseqs, dim);
    rf_rescale(retval->rf, POINTSPAN_EUC/sqrt(2));
  }
  else
    retval->rf = NULL;

  if (planar_flow == TRUE) 
    retval->pf = pf_new(retval->nseqs, dim);
  else
    retval->pf = NULL;
    
  nj_set_pointscale(retval);
  /*  retval->lambda *= retval->pointscale * retval->pointscale; */
  
  if (covar_param == CONST) {
    /* store constant */
    retval->params = vec_new(1);
    vec_set(retval->params, 0, log(max(retval->lambda-VARFLOOR, VARFLOOR)));  /* use lambda for scale; log parameterization */
  }
  else if (covar_param == DIAG) {
    retval->params = vec_new(retval->dim * retval->nseqs);
    vec_set_all(retval->params, log(max(retval->lambda-VARFLOOR, VARFLOOR)));
  }  
  else if (covar_param == DIST) {
    retval->mvn_type = MVN_GEN;
    retval->params = vec_new(1);
    vec_set(retval->params, 0, log(max(retval->lambda-VARFLOOR, VARFLOOR)));
    retval->Lapl_pinv = mat_new(dist->nrows, dist->ncols);
    retval->Lapl_pinv_evals = vec_new(dist->nrows);
    retval->Lapl_pinv_evecs = mat_new(dist->nrows, dist->nrows);
    nj_laplacian_pinv(retval);  /* set up the Laplacian pseudoinverse */    
  }
  else if (covar_param == LOWR) {
    double sdev;
    int i, j;
    
    retval->lowrank = rank;
    retval->mvn_type = MVN_LOWR;
    retval->params = vec_new(retval->lowrank * retval->nseqs);
    retval->R = mat_new(retval->nseqs, retval->lowrank);

    /* initialization is tricky; we want variances on the order of
       0.01 and expected covariances of 0 but we want to avoid
       orthogonality; initialize randomly with appropriate distrib */
    if (!seeded) {
      srandom((unsigned int)time(NULL));
      seeded = 1;
    }
    sdev = sqrt((double)LAMBDA_INIT / retval->lowrank); /* yields expected variance
                                                   of LAMBDA_INIT and expected
                                                   covariances of 0 */
    for (i = 0; i < retval->nseqs; i++) {
      for (j = 0; j < retval->lowrank; j++) {
        double draw = norm_draw(0, sdev);
        mat_set(retval->R, i, j, draw);
        vec_set(retval->params, i*retval->lowrank + j, draw);
      }
    }
  }
  else
    die("ERROR in nj_new_covar_data: unrecognized type.\n");

  return (retval);
}

void nj_dump_covar_data(CovarData *data, FILE *F) {
  fprintf(F, "CovarData\nnseqs: %d\ndim: %d\nlambda: %f\n", data->nseqs, data->dim, data->lambda);
  fprintf(F, "distance matrix:\n");
  mat_print(data->dist, F);
  fprintf(F, "Free parameters: ");
  vec_print(data->params, F);
  if (data->type == DIST) {
    fprintf(F, "Laplacian pseudoinverse:\n");
    mat_print(data->Lapl_pinv, F);
    fprintf(F, "Eigenvalues:\n");
    vec_print(data->Lapl_pinv_evals, F);
    fprintf(F, "Eigenvectors:\n");
    mat_print(data->Lapl_pinv_evecs, F);
  }
  else if (data->type == LOWR) {
    fprintf(F, "Low-rank matrix R (rank %d):\n", data->lowrank);
    mat_print(data->R, F);
  }
}

/* define Laplacian pseudoinverse from distance matrix, for use with
   DIST parameterization of covariance.  Also compute
   eigendecomposition for use in gradient calculations. Store
   everything in CovarData object */
void nj_laplacian_pinv(CovarData *data) {
  int i, dim = data->dist->nrows;
  double epsilon, trace;
  
  /* define Laplacian pseudoinverse as double centered version of
     distance matrix */
  mat_double_center(data->Lapl_pinv, data->dist, TRUE);
  
  if (mat_diagonalize_sym(data->Lapl_pinv, data->Lapl_pinv_evals, data->Lapl_pinv_evecs) != 0)
    die("ERROR in nj_laplacian_pinv: diagonalization failed.\n");    

  /* the matrix is only defined up to a translation because it is
     based on pairwise distances.  For it to define a valid covariance
     matrix (up to a positive scale constant) we need to ensure that
     it is positive definite.  We can do this by adding epsilon * I to
     it, where epsilon is equal to -1 * the smallest eigenvalue plus a
     small margin.  This will preserve the eigenvectors but shift all
     eigenvalues upward by epsilon */
  double min_eval = vec_get(data->Lapl_pinv_evals, 0);
  /* mat_diagonalize_sym guarantees eigenvalues are in ascending order */
  if (min_eval < 0) {
    epsilon = -min_eval + 1e-6;

    for (i = 0; i < dim; i++) {
      mat_set(data->Lapl_pinv, i, i, mat_get(data->Lapl_pinv, i, i) + epsilon);
      vec_set(data->Lapl_pinv_evals, i, vec_get(data->Lapl_pinv_evals, i) + epsilon);
    }
  }

  /* finally, rescale so average diagonal element is one (in space of
     final sigma), putting matrix on the same scale as the identity
     matrix used for the CONST parameterization */
  trace = vec_sum(data->Lapl_pinv_evals);
  mat_scale(data->Lapl_pinv, data->nseqs / trace);
  vec_scale(data->Lapl_pinv_evals, data->nseqs / trace);
}

