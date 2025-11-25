/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef PLANAR_FLOW_H
#define PLANAR_FLOW_H

#include <stdio.h>
#include <math.h>
#include "phast/vector.h"

#define PF_EPS 1.0e-6

/* Planar flow:
   y = x + u * tanh(w^T x + b)
   J = I + u * psi^T,  psi = (1 - tanh(s)^2) * w,  s = w^T x + b
   log|det J| = log|1 + u^T psi|
*/
typedef struct {
  int npoints;      /* number of points (taxa) */
  int ndim;         /* embedding dimension per point */

  /* parameters (shared across points) */
  Vector *u;        /* length ndim */
  Vector *w;        /* length ndim */
  double b;         /* scalar bias */

  /* gradients */
  Vector *u_grad;   /* length ndim */
  Vector *w_grad;   /* length ndim */
  double b_grad;

} PlanarFlow;

PlanarFlow *pf_new(int npoints, int ndim);
void pf_free(PlanarFlow *pf);

/* forward: y = f(x); returns log|det J| */
double pf_forward(PlanarFlow *pf, Vector *y, Vector *x);

/* backprop:
   - origgrad: dL/dy (length npoints*ndim)
   - newgrad:  populated with dL/dx
   - also fills pf->{u,w,b}_grad (accumulated across points)
*/
void pf_backprop(PlanarFlow *pf, Vector *x, Vector *newgrad, Vector *origgrad);

#endif /* PLANAR_FLOW_H */
