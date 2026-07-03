/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
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

typedef struct mixture_MVN mixture_MVN;

#define NHUTCH_SAMPLES 10  /* number of probe vectors for Hutchinson's
                              estimator of trace of Hessian */

/* caps to prevent extreme values on trace term */
#define TAYLOR_HVP_NORM_CAP 1.0e4

typedef struct taylor_data {
  struct cvdat *covar_data;
  Vector *base_grad; /* base branch-length gradient; computed
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
  Vector *y;          /* post-flow embedding at the mean (or just the
                         mean if no flows are active) */
  Vector *x;          /* pre-flow embedding at the mean (== y when no
                         flows); needed by tay_dx_from_dt so the flow
                         backprops get the correct input vector */
  struct neigh_struc *nb;
  multi_MVN *mmvn;
  TreeModel *mod;
  int component;        /* mixture component for component-specific flows */

  /* scheduling */
  double T_cache;
  int cache_ncomponents;
  double *T_cache_components;
  double *elbo_bias_components;
  unsigned int *mig_active_last_refresh_components;
  unsigned int *component_cache_initialized;
  Vector **siggrad_cache_components;
  Vector **nuis_bias_cache_components;
  unsigned int mig_active_last_refresh; /* whether migration was
                                           active when T_cache was
                                           last refreshed; if this
                                           differs from the current
                                           migration state we force
                                           a refresh to avoid a stale
                                           warmup-era residual */
  double elbo_bias;  /* EMA of (Taylor ELBO - MC ELBO), for debiasing */
  Vector *siggrad_cache;   /* size = nsigma (or full grad layout if you include mu) */
  Vector *nuis_bias_cache; /* M2: EMA of (MC nuisance grad - mean-point nuisance
                              grad), used in latent-clock mode to give the
                              per-branch rate nuisances a distribution-averaged
                              gradient every iteration instead of the biased
                              single-mean-tree estimate (size = nuis_grad) */
  int component_last_refresh; /* mixture component used for cached Taylor correction */
  int iter;    /* current iteration */
  int warmup;  /* number of warmup iterations */
  int period;  /* period between updates */
  double beta; /* for averaging of T estimates */
} TaylorData;

TaylorData *tay_new(struct cvdat *data);

void tay_free(TaylorData *td);

double elbo_taylor(TreeModel *mod, multi_MVN *mmvn, int component,
                      struct cvdat *data, Vector *grad, Vector *nuis_grad,
                      double *lprior, double *migll, double *ll_at_mean);

void tay_HVP(Vector *out, Vector *v, void *data_vd);

void tay_SVP(Vector *out, Vector *v, void *data_vd);

void tay_prep_jacobians(TaylorData *data, TreeModel *mod, Vector *x_mean);

void tay_dx_from_dt(Vector *dL_dt, Vector *dL_dx, TreeModel *mod,
                    TaylorData *data);

void tay_sigma_vec_mult(Vector *out, multi_MVN *mmvn, Vector *v,
                        struct cvdat *data, int component);

void tay_sigma_grad_mult(Vector *out, Vector *p, Vector *q,
                         multi_MVN *mmvn, struct cvdat *data, int component);

void tay_JTfun(Vector *out, Vector *v, void *userdata);

void tay_Sigmafun(Vector *out, Vector *v, void *userdata);

void tay_SigmaGradfun(Vector *grad_sigma, Vector *p_lat, Vector *q_lat,
                      void *userdata);

double elbo_hybrid(TreeModel *mod, mixture_MVN *mixmvn, int component,
                      struct cvdat *data,
                      int nminibatch, Vector *grad, Vector *nuis_grad,
                      double *lprior, double *migll, double *ll_at_mean);

#endif /* TAYLOR_H */
