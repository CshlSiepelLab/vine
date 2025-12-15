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

#ifndef TAYLOR_H
#define TAYLOR_H

#include <stdio.h>
#include <limits.h>
#include <phast/tree_model.h>
#include <covariance.h>
#include <multi_mvn.h>
#include <nj.h>

#define NHUTCH_SAMPLES 10  /* number of probe vectors for Hutchinson's
                              estimator of trace of Hessian */

typedef struct taylor_data {
  struct cvdat *covar_data;
  Vector *base_grad; /* base branch-length gradient; copmputed
                        elsewhere but copy is stored here */

  /* dimensionality; these are redundant with covar_data but
     convenient to have here */
  int nseqs;     /* number of sequences */
  int nbranches; /* number of branches in rooted tree */
  int dim;       /* embedding dimension */
  int fulld;     /* full embedding data dimension = nseqs * dim */
  int ndist;     /* number of pairwise distances = nseqs * (nseqs-1) / 2 */

    /* essential workspace vectors */
  Matrix *Jbx;    /* dim nbranches x nx */
  Matrix *JbxT;   /* dim fulld x nbranches */
  Vector *tmp_x1;    /* dim fulld */
  Vector *tmp_x2;    /* dim fulld */
  Vector *tmp_dD;    /* dim ndist */
  Vector *tmp_dy;    /* dim fulld */

  /* only needed if flows are enabled */
  Vector *tmp_extra; /* fulld */

  /* additional auxiliary data */
  Vector *y;  
  struct neigh_struc *nb;
  multi_MVN *mmvn;
  TreeModel *mod;

  /* scheduling */
  double T_cache;
  Vector *siggrad_cache;   /* size = nsigma (or full grad layout if you include mu) */
  int iter;    /* current iteration */
  int warmup;  /* number of warmup iterations */
  int period;  /* period between updates */
  double beta; /* for averaging of T estimates */
} TaylorData;

TaylorData *tay_new(struct cvdat *data);

void tay_free(TaylorData *td);

double nj_elbo_taylor(TreeModel *mod, multi_MVN *mmvn, struct cvdat *data,
                      Vector *grad, Vector *nuis_grad, double *lprior, double *migll);

void tay_HVP(Vector *out, Vector *v, void *data_vd);

void tay_SVP(Vector *out, Vector *v, void *data_vd);

void tay_prep_jacobians(TaylorData *data, TreeModel *mod, Vector *x_mean);

void tay_dx_from_dt(Vector *dL_dt, Vector *dL_dx, TreeModel *mod,
                    TaylorData *data);

void tay_sigma_vec_mult(Vector *out, multi_MVN *mmvn, Vector *v,
                        struct cvdat *data);

void tay_sigma_grad_mult(Vector *out, Vector *p, Vector *q,
                         multi_MVN *mmvn, struct cvdat *data);

void tay_JTfun(Vector *out, Vector *v, void *userdata);

void tay_Sigmafun(Vector *out, Vector *v, void *userdata);

void tay_SigmaGradfun(Vector *grad_sigma, Vector *p_lat, Vector *q_lat,
                      void *userdata);

#endif /* TAYLOR_H */
