/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Euclidean gauge fixing for embedding mixtures.

   The phylogenetic decoder depends only on pairwise Euclidean distances,
   making translation, rotation, and reflection of the latent embedding
   non-identifiable. These helpers remove this gauge freedom by expressing
   all mixture components in a common Euclidean coordinate system without
   changing the decoded pairwise distances. This procedure is only valid
   for covariance parameterizations that remain invariant under orthogonal
   changes of basis. */

#include <assert.h>
#include <math.h>
#include <gauge_fixing.h>
#include <phast/eigen.h>
#include <phast/misc.h>

/* Remove the translational gauge freedom by centering a Euclidean
   embedding at the origin. The centroid is returned in `center`
   so that associated model parameters (e.g. flow transforms) can
   be updated consistently. Pairwise distances are unchanged. */
static void center_euclidean_embedding(Vector *x, int n, int dim,
                                       Vector *center) {
  int i, d;

  assert(x->size == n * dim);
  assert(center->size == dim);
  vec_zero(center);
  for (i = 0; i < n; i++)
    for (d = 0; d < dim; d++)
      vec_set(center, d, vec_get(center, d) + vec_get(x, i*dim + d));
  vec_scale(center, 1.0 / n);

  for (i = 0; i < n; i++)
    for (d = 0; d < dim; d++)
      vec_set(x, i*dim + d, vec_get(x, i*dim + d) - vec_get(center, d));
}

/* Apply an orthogonal change of basis to a collection of Euclidean
   row vectors. The transformation preserves Euclidean distances and
   is used to align embeddings, gradients, and flow parameters to a
   common coordinate system. */
static void rotate_euclidean_vectors(Vector *x, int nrows, int dim,
                                     Matrix *Q) {
  int row, a, b;
  Vector *rot = vec_new(x->size);

  assert(x->size == nrows * dim);
  assert(Q->nrows == dim && Q->ncols == dim);
  for (row = 0; row < nrows; row++) {
    for (b = 0; b < dim; b++) {
      double val = 0.0;
      for (a = 0; a < dim; a++)
        val += vec_get(x, row*dim + a) * mat_get(Q, a, b);
      vec_set(rot, row*dim + b, val);
    }
  }

  vec_copy(x, rot);
  vec_free(rot);
}

/* Compute the orthogonal Procrustes transformation that best aligns
   the centered embedding `x` to the centered reference embedding.
   Returns TRUE if a numerically stable rotation/reflection matrix
   could be computed and stores it in `Q`. */
static int compute_procrustes_rotation(Matrix *Q, Vector *x, Vector *ref,
                                       int n, int dim) {
  Matrix *C = mat_new(dim, dim), *CtC = mat_new(dim, dim),
    *evecs = mat_new(dim, dim);
  Vector *evals = vec_new(dim);
  int i, a, b, l, m, ok = TRUE;
  const double eps = 1.0e-12;

  mat_zero(C);
  mat_zero(CtC);
  for (i = 0; i < n; i++)
    for (a = 0; a < dim; a++)
      for (b = 0; b < dim; b++)
        mat_set(C, a, b, mat_get(C, a, b) +
                vec_get(x, i*dim + a) * vec_get(ref, i*dim + b));

  for (a = 0; a < dim; a++)
    for (b = 0; b < dim; b++) {
      double val = 0.0;
      for (m = 0; m < dim; m++)
        val += mat_get(C, m, a) * mat_get(C, m, b);
      mat_set(CtC, a, b, val);
    }

  if (mat_diagonalize_sym(CtC, evals, evecs) != 0)
    ok = FALSE;
  else {
    for (l = 0; l < dim; l++)
      if (vec_get(evals, l) <= eps)
        ok = FALSE;
  }

  if (ok) {
    mat_zero(Q);
    for (a = 0; a < dim; a++)
      for (b = 0; b < dim; b++) {
        double val = 0.0;
        for (m = 0; m < dim; m++)
          for (l = 0; l < dim; l++)
            val += mat_get(C, a, m) * mat_get(evecs, m, l) *
                   mat_get(evecs, b, l) / sqrt(vec_get(evals, l));
        mat_set(Q, a, b, val);
      }
  }

  mat_free(C);
  mat_free(CtC);
  mat_free(evecs);
  vec_free(evals);
  return ok;
}

/* Update normalizing-flow parameters after the embedding basis has
   been translated and, optionally, rotated. This keeps the flow
   transformation equivalent in the new Euclidean coordinate system
   without changing the represented density. */
static void transform_flows_to_euclidean_basis(CovarData *data, int component,
                                               Vector *center, Matrix *Q,
                                               int rotate) {
  int d;

  if (data->rfs[component] != NULL) {
    for (d = 0; d < data->dim; d++)
      vec_set(data->rfs[component]->ctr, d,
              vec_get(data->rfs[component]->ctr, d) - vec_get(center, d));
    if (rotate) {
      rotate_euclidean_vectors(data->rfs[component]->ctr, 1, data->dim, Q);
      if (data->rfs[component]->ctr_grad != NULL)
        rotate_euclidean_vectors(data->rfs[component]->ctr_grad, 1,
                                 data->dim, Q);
    }
  }

  if (data->pfs[component] != NULL) {
    double bshift = 0.0;
    for (d = 0; d < data->dim; d++)
      bshift += vec_get(center, d) * vec_get(data->pfs[component]->w, d);
    data->pfs[component]->b += bshift;

    if (rotate) {
      rotate_euclidean_vectors(data->pfs[component]->u, 1, data->dim, Q);
      rotate_euclidean_vectors(data->pfs[component]->w, 1, data->dim, Q);
      if (data->pfs[component]->u_grad != NULL)
        rotate_euclidean_vectors(data->pfs[component]->u_grad, 1,
                                 data->dim, Q);
      if (data->pfs[component]->w_grad != NULL)
        rotate_euclidean_vectors(data->pfs[component]->w_grad, 1,
                                 data->dim, Q);
    }
  }
}

/* Construct a reference Euclidean embedding used to define a common
   coordinate system across mixture components. The first component is
   centered and stored as the alignment target. Returns NULL when gauge
   fixing is unnecessary or unsupported (e.g. a single component or
   hyperbolic embeddings). */
Vector *gauge_fixing_new_euclidean_reference(mixture_MVN *mixmvn,
                                             CovarData *data) {
  Vector *reference_mu, *center;

  if (mixmvn->ncomponents <= 1 || data->hyperbolic)
    return NULL;

  reference_mu = vec_new(data->nseqs * data->dim);
  center = vec_new(data->dim);
  mmvn_save_mu(mixmvn->components[0], reference_mu);
  center_euclidean_embedding(reference_mu, data->nseqs, data->dim, center);
  vec_free(center);
  return reference_mu;
}

/* Remove Euclidean gauge freedom from every mixture component by
   centering each embedding and, when compatible with the covariance
   parameterization, aligning it to the reference embedding using an
   orthogonal Procrustes transformation. Associated flow parameters and
   cached moments are transformed into the same coordinate basis. */
void gauge_fixing_apply_euclidean(mixture_MVN *mixmvn, CovarData *data,
                                  Vector *reference_mu, Vector **mu_moment) {
  int k;

  /* DIAG covariance is not compatible with this gauge fixing approach */
  int rotate = data->type == CONST || data->type == DIST || data->type == LOWR;

  if (reference_mu == NULL)
    return;

  for (k = 0; k < mixmvn->ncomponents; k++) {
    Vector *mu = vec_new(data->nseqs * data->dim);
    Vector *center = vec_new(data->dim);
    Matrix *Q = mat_new(data->dim, data->dim);
    int do_rotate = FALSE;

    mmvn_save_mu(mixmvn->components[k], mu);
    center_euclidean_embedding(mu, data->nseqs, data->dim, center);
    if (rotate && compute_procrustes_rotation(Q, mu, reference_mu,
                                              data->nseqs, data->dim)) {
      rotate_euclidean_vectors(mu, data->nseqs, data->dim, Q);
      if (mu_moment != NULL && mu_moment[k] != NULL)
        rotate_euclidean_vectors(mu_moment[k], data->nseqs, data->dim, Q);
      do_rotate = TRUE;
    }
    mmvn_set_mu(mixmvn->components[k], mu);
    transform_flows_to_euclidean_basis(data, k, center, Q, do_rotate);

    mat_free(Q);
    vec_free(center);
    vec_free(mu);
  }
}
