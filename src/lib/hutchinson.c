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

double hutch_tr(HVP_fun   Hfun,
                SVP_fun   Sfun,
                void     *data, /* auxiliary data for Hfun and Sfun */
                int       dim,
                int       nprobe)
{
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
