/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* KL divergence metrics between two posterior tree samples */

#include <math.h>
#include <phast/lists.h>
#include <phast/trees.h>
#include <phast/misc.h>
#include <phast/stringsplus.h>
#include <nj.h>
#include <geometry.h>
#include <multi_mvn.h>
#include <rf.h>
#include <vine.h>
#include "kl.h"


void tr_split_kl(List *trees_est, List *trees_ref, double *mean_kl) {
  *mean_kl = 0.0;
  int S_est = lst_size(trees_est), S_ref = lst_size(trees_ref);
  if (S_est == 0 || S_ref == 0) return;

  int nsc_e, nsc_r, n_e, n_r;
  SplitCount *sce = tr_collect_split_counts(trees_est, &nsc_e, &n_e);
  SplitCount *scr = tr_collect_split_counts(trees_ref, &nsc_r, &n_r);

  if (n_e != n_r)
    die("ERROR in tr_split_kl: estimate and reference trees have different numbers of leaves.\n");

  int i = 0, j = 0, nsplits = 0;
  double sum_kl = 0.0;
  while (i < nsc_e || j < nsc_r) {
    int cmp;
    if (i >= nsc_e) cmp = 1;
    else if (j >= nsc_r) cmp = -1;
    else cmp = tr_bitmask_cmp(sce[i].mask, scr[j].mask);

    int c_e = (cmp <= 0) ? sce[i].count : 0;
    int c_r = (cmp >= 0) ? scr[j].count : 0;

    /* Laplace-smoothed inclusion probabilities so no split has p or q
       exactly 0 or 1 (which would blow up the KL divergence below). */
    double p = (c_e + 0.5) / (S_est + 1.0);
    double q = (c_r + 0.5) / (S_ref + 1.0);
    double kl = p * log(p / q) + (1.0 - p) * log((1.0 - p) / (1.0 - q));
    sum_kl += kl;
    nsplits++;

    if (cmp <= 0) i++;
    if (cmp >= 0) j++;
  }

  *mean_kl = (nsplits > 0) ? sum_kl / nsplits : 0.0;

  for (int k = 0; k < nsc_e; k++) tr_bitmask_free(sce[k].mask);
  for (int k = 0; k < nsc_r; k++) tr_bitmask_free(scr[k].mask);
  sfree(sce); sfree(scr);
}

/* ---- helpers for tr_embed_kl ---------------------------------------------- */

/* Embed one tree's leaves into a dim-dimensional Euclidean space via
   classical MDS on its cophenetic distances (the same closed-form
   procedure vine.c's --dist-embedding option uses), then return the
   n*(n-1)/2 pairwise Euclidean distances between the embedded leaves,
   in (i<j) order matching names[]/tree_to_distances. Caller frees the
   returned Vector. */
static Vector *embed_tree_pairwise_dists(TreeNode *tree, char **names, int n, int dim) {
  Matrix *D = tree_to_distances(tree, names, n);
  CovarData *data = new_covar_data(CONST, D, dim, NULL, NULL, names,
                                    FALSE, 1.0, 3, 1.0, FALSE, 1.0, FALSE,
                                    FALSE, FALSE, 1, NULL, NULL, FALSE);
  multi_MVN *mmvn = mmvn_new(n, dim, MVN_DIAG);
  estimate_mmvn_from_distances(data, mmvn, 0);

  /* for MVN_DIAG, embedding coordinates are stored flat in mmvn->mvn->mu,
     laid out as point i's k-th coordinate at index i*dim + k (matching
     estimate_mmvn_from_distances_euclidean's mu_full layout); mmvn->mu[]
     is only populated for MVN_GEN/MVN_LOWR. */
  Vector *mu = mmvn->mvn->mu;
  Vector *dists = vec_new(n * (n - 1) / 2);
  int idx = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double ssq = 0.0;
      for (int k = 0; k < dim; k++) {
        double diff = vec_get(mu, i*dim + k) - vec_get(mu, j*dim + k);
        ssq += diff * diff;
      }
      vec_set(dists, idx++, sqrt(ssq));
    }
  }

  mmvn_free(mmvn);
  free_covar_data(data);   /* also frees D */
  return dists;
}

/* accumulate per-pair (mean, sd) of embedded pairwise distances across
   a set of trees. names/n define the fixed leaf order shared by both
   estimate and reference sets. Caller frees the returned arrays. */
static void embed_pair_stats(List *trees, char **names, int n, int dim,
                              double **mean_out, double **sd_out) {
  int S = lst_size(trees);
  int npairs = n * (n - 1) / 2;
  double *sum = calloc(npairs, sizeof(double));
  double *sumsq = calloc(npairs, sizeof(double));

  for (int s = 0; s < S; s++) {
    TreeNode *t = lst_get_ptr(trees, s);
    Vector *d = embed_tree_pairwise_dists(t, names, n, dim);
    for (int k = 0; k < npairs; k++) {
      double v = vec_get(d, k);
      sum[k] += v;
      sumsq[k] += v * v;
    }
    vec_free(d);
  }

  double *mean = smalloc(npairs * sizeof(double));
  double *sd = smalloc(npairs * sizeof(double));
  for (int k = 0; k < npairs; k++) {
    mean[k] = sum[k] / S;
    double var = sumsq[k] / S - mean[k] * mean[k];
    if (var < 1e-12) var = 1e-12;   /* floor to avoid degenerate KL below */
    sd[k] = sqrt(var);
  }
  free(sum); free(sumsq);

  *mean_out = mean;
  *sd_out = sd;
}

void tr_embed_kl(List *trees_est, List *trees_ref, int dim, double *mean_kl) {
  *mean_kl = 0.0;
  int S_est = lst_size(trees_est), S_ref = lst_size(trees_ref);
  if (S_est == 0 || S_ref == 0) return;

  TreeNode *t0 = lst_get_ptr(trees_est, 0);
  List *namelist = tr_leaf_names(t0);
  lst_qsort_str(namelist, ASCENDING);
  int n = lst_size(namelist);
  if (n < 3) { lst_free_strings(namelist); lst_free(namelist); return; }

  char **names = smalloc(n * sizeof(char*));
  for (int i = 0; i < n; i++)
    names[i] = ((String*)lst_get_ptr(namelist, i))->chars;

  if (dim <= 0)
    dim = (int) round(DEFAULT_DIM_INTERCEPT + DEFAULT_DIM_SLOPE * log((double)n));
  if (dim < 1) dim = 1;

  double *mean_e, *sd_e, *mean_r, *sd_r;
  embed_pair_stats(trees_est, names, n, dim, &mean_e, &sd_e);
  embed_pair_stats(trees_ref, names, n, dim, &mean_r, &sd_r);

  int npairs = n * (n - 1) / 2;
  double sum_kl = 0.0;
  for (int k = 0; k < npairs; k++) {
    double kl = log(sd_r[k] / sd_e[k])
              + (sd_e[k]*sd_e[k] + (mean_e[k]-mean_r[k])*(mean_e[k]-mean_r[k])) / (2.0*sd_r[k]*sd_r[k])
              - 0.5;
    sum_kl += kl;
  }
  *mean_kl = (npairs > 0) ? sum_kl / npairs : 0.0;

  sfree(names);
  lst_free_strings(namelist);
  lst_free(namelist);
  free(mean_e); free(sd_e); free(mean_r); free(sd_r);
}
