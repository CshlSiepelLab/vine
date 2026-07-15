/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Robinson-Foulds distance and tree-sample entropy metrics */

#include <stdlib.h>
#include <math.h>
#include <phast/misc.h>
#include "tree_splits.h"
#include "rf.h"

double tr_robinson_foulds(TreeNode *t1, TreeNode *t2) {
  TreeSplitContext *ctx = tr_split_context_new(t1);
  if (tr_split_context_nleaves(ctx) < 3) {
    tr_split_context_free(ctx);
    return 0.0;
  }

  TreeSplitVector *s1 = tr_collect_splits(ctx, t1, FALSE);
  TreeSplitVector *s2 = tr_collect_splits(ctx, t2, FALSE);
  int i = 0, j = 0, common = 0;
  while (i < s1->size && j < s2->size) {
    int cmp = bs_compare(s1->splits[i].mask, s2->splits[j].mask);
    if (cmp == 0) {
      common++;
      i++;
      j++;
    }
    else if (cmp < 0)
      i++;
    else
      j++;
  }

  int distance = s1->size + s2->size - 2 * common;
  tr_split_vector_free(s1);
  tr_split_vector_free(s2);
  tr_split_context_free(ctx);
  return distance;
}

Matrix *tr_robinson_foulds_matrix(List *trees, unsigned int log_progress) {
  int ntrees = lst_size(trees);
  Matrix *matrix = mat_new(ntrees, ntrees);

  for (int i = 0; i < ntrees; i++) {
    mat_set(matrix, i, i, 0.0);
    for (int j = i + 1; j < ntrees; j++) {
      double d = tr_robinson_foulds(lst_get_ptr(trees, i),
                                    lst_get_ptr(trees, j));
      mat_set(matrix, i, j, d);
      mat_set(matrix, j, i, 0.0);
    }
    if (log_progress && (i + 1) % 100 == 0)
      fprintf(stderr, "Computed RF distances for %d of %d trees...\n",
              i + 1, ntrees);
  }
  return matrix;
}

void tr_write_robinson_foulds_matrix(Matrix *matrix, FILE *F) {
  if (matrix->nrows != matrix->ncols)
    die("ERROR: Robinson-Foulds distance matrix must be square.\n");

  fprintf(F, "tree");
  for (int i = 0; i < matrix->ncols; i++)
    fprintf(F, "\ttree%d", i + 1);
  fprintf(F, "\n");
  for (int i = 0; i < matrix->nrows; i++) {
    fprintf(F, "tree%d", i + 1);
    for (int j = 0; j < matrix->ncols; j++)
      fprintf(F, "\t%.0f", mat_get(matrix, i, j));
    fprintf(F, "\n");
  }
}

typedef struct {
  TreeSplitVector *vec;
} TreeSplitData;

/* Global state for qsort topology comparison. */
static TreeSplitData *g_tsd;

static int compare_tree_indices(const void *pa, const void *pb) {
  const TreeSplitVector *a = g_tsd[*(const int *)pa].vec;
  const TreeSplitVector *b = g_tsd[*(const int *)pb].vec;
  int size = a->size < b->size ? a->size : b->size;
  for (int i = 0; i < size; i++) {
    int cmp = bs_compare(a->splits[i].mask, b->splits[i].mask);
    if (cmp != 0)
      return cmp;
  }
  return a->size - b->size;
}

static int same_topology(const TreeSplitVector *a, const TreeSplitVector *b) {
  if (a->size != b->size)
    return FALSE;
  for (int i = 0; i < a->size; i++)
    if (bs_compare(a->splits[i].mask, b->splits[i].mask) != 0)
      return FALSE;
  return TRUE;
}

static double log_branch_length(double length) {
  return log(length > 0.0 ? length : 1e-300);
}

void tr_tree_entropy(List *trees, double *H_split, double *H_top,
                     double *mean_var, double *mean_var_per_branch) {
  int ntrees = lst_size(trees);
  *H_split = 0.0;
  *H_top = 0.0;
  *mean_var = 0.0;
  *mean_var_per_branch = 0.0;
  if (ntrees == 0)
    return;

  TreeSplitContext *ctx = tr_split_context_new(lst_get_ptr(trees, 0));
  if (tr_split_context_nleaves(ctx) < 3) {
    tr_split_context_free(ctx);
    return;
  }

  /* Split entropy from posterior inclusion frequencies. */
  int ncounts;
  SplitCount *counts = tr_collect_split_counts(ctx, trees, &ncounts);
  for (int i = 0; i < ncounts; i++) {
    double p = (double)counts[i].count / ntrees;
    if (p > 0.0 && p < 1.0)
      *H_split += -p * log(p) - (1.0 - p) * log(1.0 - p);
  }
  tr_split_counts_free(counts, ncounts);

  /* Collect all edges for topology grouping and branch-length variance. */
  TreeSplitData *tsd = smalloc(ntrees * sizeof(TreeSplitData));
  for (int i = 0; i < ntrees; i++)
    tsd[i].vec = tr_collect_splits(ctx, lst_get_ptr(trees, i), TRUE);

  int *order = smalloc(ntrees * sizeof(int));
  for (int i = 0; i < ntrees; i++)
    order[i] = i;
  g_tsd = tsd;
  qsort(order, ntrees, sizeof(int), compare_tree_indices);

  int group_start = 0;
  while (group_start < ntrees) {
    int group_end = group_start + 1;
    TreeSplitVector *first = tsd[order[group_start]].vec;
    while (group_end < ntrees &&
           same_topology(first, tsd[order[group_end]].vec))
      group_end++;

    int group_size = group_end - group_start;
    int nbranches = first->size;
    double topology_prob = (double)group_size / ntrees;
    *H_top += -topology_prob * log(topology_prob);

    if (group_size >= 2 && nbranches > 0) {
      double *mean = calloc(nbranches, sizeof(double));
      for (int i = group_start; i < group_end; i++) {
        TreeSplitVector *vec = tsd[order[i]].vec;
        for (int k = 0; k < nbranches; k++)
          mean[k] += log_branch_length(vec->splits[k].blen);
      }
      for (int k = 0; k < nbranches; k++)
        mean[k] /= group_size;

      double variance_sum = 0.0;
      for (int k = 0; k < nbranches; k++) {
        double variance = 0.0;
        for (int i = group_start; i < group_end; i++) {
          TreeSplitVector *vec = tsd[order[i]].vec;
          double diff = log_branch_length(vec->splits[k].blen) - mean[k];
          variance += diff * diff;
        }
        variance_sum += variance / (group_size - 1);
      }
      *mean_var += topology_prob * variance_sum;
      free(mean);
    }
    group_start = group_end;
  }

  int nbranches = tsd[order[0]].vec->size;
  *mean_var_per_branch = nbranches > 0 ? *mean_var / nbranches : 0.0;

  sfree(order);
  for (int i = 0; i < ntrees; i++)
    tr_split_vector_free(tsd[i].vec);
  sfree(tsd);
  tr_split_context_free(ctx);
}
