/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef GEO_H
#define GEO_H

#include <stdio.h>
#include <limits.h>
#include <phast/matrix.h>
#include <phast/msa.h>
#include <phast/trees.h>
#include <phast/tree_model.h>
#include <nj.h>
#include <mvn.h>
#include <multi_mvn.h>
#include <sparse_matrix.h>
#include <crispr.h>
#include <phast/misc.h>
#include <radial_flow.h>
#include <planar_flow.h>
#include <tree_prior.h>
#include <migration.h>

/* rescale embedding space so that median distances are equal to
   these values in the Euclidean and hyperbolic cases,
   respectively. Helps address the problem that branch lengths tend to
   be small so means and variances can get close to zero.  But scaling
   needs to be different for the two geometries */
#define POINTSPAN_EUC 25 
#define POINTSPAN_HYP 4

void nj_points_to_distances(Vector *points, CovarData *data);

void nj_points_to_distances_euclidean(Vector *points, CovarData *data);

void nj_points_to_distances_hyperbolic(Vector *points, CovarData *data);

void nj_estimate_mmvn_from_distances(CovarData *data, multi_MVN *mmvn);

void nj_estimate_mmvn_from_distances_euclidean(CovarData *data, multi_MVN *mmvn);

void nj_estimate_mmvn_from_distances_hyperbolic(CovarData *data, multi_MVN *mmvn);

void nj_mmvn_to_distances(multi_MVN *mmvn, CovarData *data);

void nj_test_D(Matrix *D);

void nj_set_pointscale(CovarData *data);


/* these are used for the hyperbolic geometry to stabilize the acosh
   calculations */
/* Same thresholds in BOTH places (distance and gradient) */
#define ACOSH_EPS   1e-8     /* near u≈1 */
#define ACOSH_HUGE  1e+8     /* asymptotic regime */

/* Stable acosh: series near 1, log form at very large u */
static inline double acosh_stable(double u) {
  if (u < 1.0) u = 1.0;                      /* safety */
  double e = u - 1.0;
  if (e < ACOSH_EPS) {
    /* acosh(1+e) ≈ sqrt(2e) * (1 - e/12) */
    double ef = fmax(e, 1e-18);
    double rt = sqrt(2.0*ef);
    return rt * (1.0 - e/12.0);
  }
  if (u > ACOSH_HUGE) {
    /* acosh(u) ~ log(2u) with tiny relative error */
    return log(u) + log(2.0);
  }
  /* log1p reduces cancellation when u≈1 */
  return log1p((u - 1.0) + sqrt((u - 1.0) * (u + 1.0)));
}

/* Stable derivative d/du acosh(u): series near 1, asymptotic at large u */
static inline double d_acosh_du_stable(double u) {
  if (u < 1.0) u = 1.0;
  double e = u - 1.0;
  if (e < ACOSH_EPS) {
    double ef = fmax(e, 1e-18);
    /* d/du acosh(1+e) ≈ (1/√(2e)) * (1 - e/4) */
    return (1.0 / sqrt(2.0 * ef)) * (1.0 - e/4.0);
  }
  if (u > ACOSH_HUGE) {
    /* 1/√(u^2-1) ≈ (1/u) * (1 + 1/(2u^2))  for large u  */
    return (1.0 / u) * (1.0 + 1.0 / (2.0 * u * u));
  }
  return 1.0 / sqrt(u * u - 1.0);
}

#endif
