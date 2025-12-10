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
  Vector *base_grad; /* base branch-length gradient for Taylor approximation */
  Matrix *Jbx;    /* size nbranches x nx */
  Matrix *JbxT;   /* size nx x nbranches */
  int nbranches;  /* FIXME: redundant */
  int nx;    /* FIXME: redundant */
  Vector *tmp_x1, *tmp_x2, *tmp_dD, *tmp_dy, *y, *tmp_extra; /* Reusable workspace vectors */
  struct neigh_struc *nb;
  multi_MVN *mmvn;
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

void tay_sigma_vec_mult(Vector *out, multi_MVN *mmvn, Vector *v);

#endif /* TAYLOR_H */
