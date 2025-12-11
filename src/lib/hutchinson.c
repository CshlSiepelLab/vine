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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <phast/vector.h>
#include <mvn.h>
#include <hutchinson.h>

/* Compute T = tr(H S) using Hutchinson's trace estimator.

   Hfun: function to compute H v for arbitrary vector v
   Sfun: function to compute S v for arbitrary vector v
   data: auxiliary data passed to Hfun and Sfun
   dim:  dimensionality of vectors (nbranches)
   nprobe: number of Hutchinson samples to use */
double hutch_tr(HVP_fun   Hfun,
                SVP_fun   Sfun,
                void     *data, /* auxiliary data for Hfun and Sfun */
                int       dim,
                int       nprobe) {
  double accum = 0.0;

  Vector *z  = vec_new(dim);
  Vector *u  = vec_new(dim);
  Vector *Hu = vec_new(dim);

  for (int k = 0; k < nprobe; k++) {

    /* z ~ MVN(0,1) */
    mvn_sample_std(z);

    /* u = S_b z */
    Sfun(u, z, data);

    /* Hu = H u  (Pearlmutter directional derivative) */
    Hfun(Hu, u, data);

    /* Contribution = z^T (H u) */
    accum += vec_inner_prod(z, Hu);
  }

  vec_free(z);
  vec_free(u);
  vec_free(Hu);

  return accum / nprobe;
}

/* Generalization of above that optionally computes gradient also.  In
   particular, compute,

       T = tr(H S)

   and optionally also compute the gradient wrt the covariance
   parameters using:

       ∇_σ T = E_z [ ∂/∂σ (u_latᵀ Σ u_lat) ]

   Everything model-specific is passed in via function pointers.

   dim_out     = dimensionality of branch-space vectors (nbranches)
   dim_lat     = latent coordinate dimension (n * d) */
double hutch_tr_plus_grad(
                        HVP_fun        Hfun,          /* Hessian–vector product    */
                        SVP_fun        Sfun,          /* S * v                     */
                        JT_fun         JTfun,         /* latent u_lat = Jᵀ * v     */
                        Sigma_fun      Sigmafun,      /* Σ * v_lat                  */
                        SigmaGrad_fun  SigmaGradFun,  /* accumulate ∂/∂σ (vᵀ Σ v)  */

                        void          *userdata,      /* opaque pointer passed thru */

                        int            dim_out,       /* branch-space dimension     */
                        int            dim_lat,       /* latent-space dimension     */
                        int            nprobe,        /* # of Hutchinson samples    */

                        Vector        *grad_sigma     /* optional output (may be NULL) */
                        ) {
  double accum = 0.0;
    
  Vector *z      = vec_new(dim_out);
  Vector *u      = vec_new(dim_out);     /* S z           */
  Vector *Hu     = vec_new(dim_out);     /* H(S z)        */

  Vector *u_lat  = vec_new(dim_lat);     /* Jᵀ z          */
  Vector *tmp    = vec_new(dim_lat);     /* Σ u_lat       */

  for (int k = 0; k < nprobe; k++) {

    /* z ~ N(0, I) or Rademacher */
    mvn_sample_std(z);

    /* u = S z */
    Sfun(u, z, userdata);

    /* Hu = H u */
    Hfun(Hu, u, userdata);

    /* accumulate trace component: zᵀ H u */
    accum += vec_inner_prod(z, Hu);

    /* If no gradient requested, skip this part */
    if (grad_sigma == NULL)
      continue;

    /* --------- gradient wrt covariance parameters ---------- */

    /* latent-space vector: u_lat = Jᵀ z */
    JTfun(u_lat, z, userdata);

    /* tmp = Σ u_lat */
    Sigmafun(tmp, u_lat, userdata);

    /* accumulate ∂/∂σ (u_latᵀ Σ u_lat) */
    SigmaGradFun(grad_sigma, u_lat, userdata);
  }

  /* Scale gradient by 1/nprobe */
  if (grad_sigma != NULL) {
    double scale = 1.0 / nprobe;
    vec_zero(grad_sigma);
    for (int i = 0; i < grad_sigma->size; i++)
      vec_set(grad_sigma, i,
              scale * vec_get(grad_sigma, i));
  }

  vec_free(z);
  vec_free(u);
  vec_free(Hu);
  vec_free(u_lat);
  vec_free(tmp);

  /* return tr(H S) */
  return accum / nprobe;
}
