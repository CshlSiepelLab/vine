/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* calculations based on euclidean and hyperbolic geometries */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>
#include <phast/eigen.h>
/* convert an nd-dimensional vector to an nxn upper triangular
   distance matrix.  Assumes each taxon is represented as a point in
   d-dimensional space.  Wrapper for versions that assume either
   Euclidean or hyperbolic geometry */
void nj_points_to_distances(Vector *points, CovarData *data) {
  if (data->hyperbolic)
    nj_points_to_distances_hyperbolic(points, data);
  else
    nj_points_to_distances_euclidean(points, data);
}
  
/* convert an nd-dimensional vector to an nxn upper triangular
   distance matrix.  Assumes each taxon is represented as a point in
   d-dimensional space and computes Euclidean distances between these
   points */ 
void nj_points_to_distances_euclidean(Vector *points, CovarData *data) {
  int i, j, k, vidx1, vidx2, n, d;
  double sum;
  double dist, maxdist = 0;
  Matrix *D = data->dist;

  n = D->nrows;
  d = points->size / n;

  if (points->size != n*d || D->nrows != D->ncols) 
    die("ERROR nj_points_to_distances_euclidean: bad dimensions\n");

  mat_zero(D);
  for (i = 0; i < n; i++) {
    vidx1 = i*d;
    for (j = i+1; j < n; j++) {
      vidx2 = j*d;
      sum = 0;
      for (k = 0; k < d; k++) {
        double diff = vec_get(points, vidx1 + k) -
          vec_get(points, vidx2 + k);
        sum += diff*diff;
      }
      dist = sqrt(sum) / data->pointscale;
      mat_set(D, i, j, dist);
      if (dist > maxdist) {
        maxdist = dist;
        data->tree_diam_leaf1 = i; /* store these for use in rerooting, if needed */
        data->tree_diam_leaf2 = j;
      }
    }
  }
}

/* wrapper for nj_points_to_distances functions */
void nj_mmvn_to_distances(multi_MVN *mmvn, CovarData *data) {
  Vector *full_mu;
  if (mmvn->type != MVN_GEN && mmvn->type != MVN_LOWR) 
    full_mu = mmvn->mvn->mu;
  else {
    full_mu = vec_new(mmvn->d * mmvn->n);
    mmvn_save_mu(mmvn, full_mu);
  }

  nj_points_to_distances(full_mu, data);

  if (mmvn->type == MVN_GEN || mmvn->type == MVN_LOWR)
    vec_free(full_mu);
}

/* convert an nd-dimensional vector to an nxn upper triangular
   distance matrix.  Assumes each taxon is represented as a point in
   d-dimensional space and computes hyperbolic distances between these
   points */ 
void nj_points_to_distances_hyperbolic(Vector *points, CovarData *data) {
  int i, j, k, vidx1, vidx2, n, d;
  double lor_inner, ss1, ss2, x0_1, x0_2, Dij, u, maxdist = 0;
  Matrix *D = data->dist;
  double alpha = 1.0 / sqrt(data->negcurvature);   /* curvature radius */

  n = D->nrows;
  d = points->size / n;
  
  if (points->size != n*d || D->nrows != D->ncols) {
    die("ERROR nj_points_to_distances_hyperbolic: bad dimensions\n");
  }

  mat_zero(D);
  for (i = 0; i < n; i++) {
    vidx1 = i*d;
    for (j = i+1; j < n; j++) {
      vidx2 = j*d;
      lor_inner = 0;
      ss1 = 1;
      ss2 = 1;
      for (k = 0; k < d; k++) {
        double xi = vec_get(points, vidx1 + k);
        double xj = vec_get(points, vidx2 + k);
        lor_inner += xi * xj ;
        ss1 += xi * xi;
        ss2 += xj * xj;
      }
      x0_1 = sqrt(ss1); /* the 0th dimension for each point is determined by the
                           others, to stay on the hyperboloid */
      x0_2 = sqrt(ss2);

      lor_inner -= x0_1 * x0_2;  /* last term of Lorentz inner product */

      u = -lor_inner;
      Dij = (alpha / data->pointscale) * acosh_stable(u);

      assert(isfinite(Dij) && Dij >= 0);

      mat_set(D, i, j, Dij);      

      if (Dij > maxdist) {
        maxdist = Dij;
        data->tree_diam_leaf1 = i; /* store these for use in rerooting, if needed */
        data->tree_diam_leaf1 = j;
      }
    }
  }
}

/* generate an approximate multivariate normal distribution from a
   distance matrix, for use in initializing the variational inference
   algorithm.  */
void nj_estimate_mmvn_from_distances(CovarData *data, multi_MVN *mmvn) {
  if (data->hyperbolic)
    nj_estimate_mmvn_from_distances_hyperbolic(data, mmvn);
  else
    nj_estimate_mmvn_from_distances_euclidean(data, mmvn);  
}

/* generate an approximate multivariate normal distribution from a distance matrix, for
   use in initializing the variational inference algorithm. Uses multidimensional scaling  */
void nj_estimate_mmvn_from_distances_euclidean(CovarData *data, multi_MVN *mmvn) {
  Matrix *D = data->dist;
  int n = D->nrows;
  Matrix *Dsq, *G, *revec_real;
  Vector *eval_real;
  int i, j, d;
  Vector *mu_full = vec_new(data->dim * n);
  
  if (D->nrows != D->ncols || mmvn->d * mmvn->n != data->dim * n)
    die("ERROR in nj_estimate_points_from_distances: bad dimensions\n");

  /* build matrix of squared distances; note that D is upper
     triangular but Dsq must be symmetric */
  Dsq = mat_new(n, n);
  for (i = 0; i < n; i++) {
    mat_set(Dsq, i, i, 0);
    for (j = i + 1; j < n; j++) {
      double d2 = mat_get(D, i, j) * mat_get(D, i, j);
      mat_set(Dsq, i, j, d2);
      mat_set(Dsq, j, i, d2);
    }
  }

  /* double center */
  G = mat_new(n, n);
  mat_double_center(G, Dsq, FALSE);
  
  /* find eigendecomposition of G */
  eval_real = vec_new(n); vec_zero(eval_real);
  revec_real = mat_new(n, n); mat_zero(revec_real);
  if (mat_diagonalize_sym(G, eval_real, revec_real) != 0)
    die("ERROR in nj_estimate_mmvn_from_distances_euclidean: diagonalization failed.\n");
  
  /* create a vector of points based on the first 'dim' eigenvalues */
  for (d = 0; d < data->dim; d++) {
    double eval = vec_get(eval_real, n-1-d);
    if (eval < 0) eval = 0;
    double sqeval = sqrt(eval);
    /* product of evalsqrt and corresponding column of revec will define
       the dth component of each point */    
    for (i = 0; i < n; i++) {
      vec_set(mu_full, i*data->dim + d,
              sqeval * mat_get(revec_real, i, n-1-d));
    }
  }
  
  /* rescale */
  vec_scale(mu_full, data->pointscale);
  mmvn_set_mu(mmvn, mu_full);

  /* covariance parameters should already be initialized */
  nj_update_covariance(mmvn, data);
  
  mat_free(Dsq);
  mat_free(G);
  vec_free(eval_real);
  mat_free(revec_real);
  vec_free(mu_full);
}

/* generate an approximate mu and sigma from a distance matrix, for
   use in initializing the variational inference algorithm. In this
   version, use the 'hydra' algorithm to solve the problem
   approximately in hyperbolic space (Keller-Ressel & Nargang,
   arXiv:1903.08977, 2019) */
void nj_estimate_mmvn_from_distances_hyperbolic(CovarData *data, multi_MVN *mmvn) {
  Matrix *D = data->dist;
  int n = D->nrows;
  Matrix *A, *revec_real;
  Vector *eval_real;
  int i, j, d;
  Vector *mu_full = vec_new(data->dim*n);
    
  if (D->nrows != D->ncols || mmvn->d * mmvn->n != data->dim * n)
    die("ERROR in nj_estimate_points_from_distances_hyperbolic: bad dimensions\n");
  
  /* build matrix A of transformed distances; note that D is upper
     triangular but A must be symmetric */
  A = mat_new(n, n);
  for (i = 0; i < n; i++) {
    mat_set(A, i, i, 1);
    for (j = i + 1; j < n; j++) {
      double a = cosh(sqrt(data->negcurvature) * mat_get(D, i, j) * data->pointscale);
      mat_set(A, i, j, a);
      mat_set(A, j, i, a);
    }
  }
  
  /* find eigendecomposition of A */
  eval_real = vec_new(n);
  revec_real = mat_new(n, n);
  if (mat_diagonalize_sym(A, eval_real, revec_real) != 0)
    die("ERROR in nj_estimate_mmvn_from_distances_hyperbolic: diagonalization failed.\n");

  /* create a vector of points based on the first 'dim' eigenvalues */
  for (d = 0; d < data->dim; d++) {
    /* product of evalsqrt and corresponding column of revec will define
       the dth component of each point */
    double ev = -vec_get(eval_real, d);
    assert(isfinite(ev));
    if (ev < 0) ev = 1e-6;
    for (i = 0; i < n; i++) {
      vec_set(mu_full, i*data->dim + d, sqrt(ev) * mat_get(revec_real, i, d)); 
      assert(isfinite(vec_get(mu_full, i*data->dim + d)));
    }
  }

  mmvn_set_mu(mmvn, mu_full); 
  
  /* covariance parameters should already be initialized */
  nj_update_covariance(mmvn, data);
  
  mat_free(A);
  vec_free(eval_real);
  mat_free(revec_real);
  vec_free(mu_full);
}

/* ensure a distance matrix is square, upper triangular, has zeroes on
   main diagonal, and has all non-negative entries.  */
void nj_test_D(Matrix *D) {
  int n = D->nrows;
  int i, j;
  if (n != D->ncols)
    die("ERROR in nj_test_D: bad dimensions in distance matrix D\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (j <= i && mat_get(D, i, j) != 0)
	die("ERROR in nj_test_D: distance matrix must be upper triangular and have zeroes on main diagonal.\n");
      else if (mat_get(D, i, j) < 0 || !isfinite(mat_get(D, i, j)))
	die("ERROR in nj_test_D: entries in distance matrix must be nonnegative and finite\n");
    }
  }
}

/* set scale factor for geometry depending on starting distance matrix */
void nj_set_pointscale(CovarData *data) {
  /* find median pairwise distance */
  double medianD = mat_median_upper_triang(data->dist);  /* off-diagonal median */
  if (medianD <= 0.0 || !isfinite(medianD)) 
    data->pointscale = 1.0;   /* safe backup */
  else if (data->hyperbolic == TRUE)
    data->pointscale = POINTSPAN_HYP / (medianD * sqrt(data->negcurvature));
  else
    data->pointscale = POINTSPAN_EUC / medianD; 
}
