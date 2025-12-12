/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* approximation of trace of Hessian-vector product using Hutchinson's estimator */

#ifndef HUTCH_H
#define HUTCH_H

#include <stdio.h>
#include <stdlib.h>
#include <phast/vector.h>

/* Computes out = H v for arbitrary vector v using Pearlmutter
   directional derivative.  data = auxiliary data */
typedef void (*HVP_fun)(Vector *out, Vector *v, void *data);

/* Computes out = S v using factored optionally factored S and
   arbitrary vector .  data = auxiliary data. */
typedef void (*SVP_fun)(Vector *out, Vector *v, void *data);

/* Computes out = J^T * v   (latent-space dimension)  */
typedef void (*JT_fun)(Vector *out, Vector *v, void *userdata);

/* Computes out = Σ * v_lat  (latent-space covariance application) */
typedef void (*Sigma_fun)(Vector *out, Vector *v_lat, void *userdata);

/* Accumulates gradient wrt covariance parameters:
      grad_sigma += ∂/∂σ ( v_lat^T Σ v_lat )
   v_lat is J^T * z for a Hutchinson probe. */
typedef void (*SigmaGrad_fun)(Vector *grad_sigma,
                              Vector *p_lat,
                              Vector *q_lat,
                              void   *userdata);

/* Compute T = tr(H S) using Hutchinson's trace estimator. */
double hutch_tr(HVP_fun Hfun, SVP_fun Sfun, void *data, int dim,
                int nprobe);

/* Compute T = tr(H S) using Hutchinson's trace estimator and
   optionally compute gradient wrt covariance parameters */
double hutch_tr_plus_grad(HVP_fun Hfun, SVP_fun Sfun, JT_fun JTfun,
                          Sigma_fun Sigmafun, SigmaGrad_fun SigmaGradFun,
                          void *userdata, int dim_out, int dim_lat, 
                          int nprobe, Vector *grad_sigma); 
#endif /* HUTCH_H */
