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

typedef void (*HVP_fun)(Vector *out, Vector *v, void *data);
/* Computes out = H v for arbitrary vector v using Pearlmutter
   directional derivative.  data = auxiliary data */


typedef void (*SVP_fun)(Vector *out, Vector *v, void *data);
/* Computes out = S v using factored optionally factored S and
   arbitrary vector .  data = auxiliary data. */

double hutch_tr(HVP_fun Hfun, SVP_fun Sfun, void *data, int dim,
                int nprobe);

double hutch_tr_general(HVP_fun Hfun, SVP_fun Sfun, JT_fun JTfun,
                        Sigma_fun Sigmafun, SigmaGrad_fun SigmaGradFun,
                        void *userdata, int dim_out, int dim_lat, 
                        int nprobe, Vector *grad_sigma_opt); 
#endif /* HUTCH_H */
