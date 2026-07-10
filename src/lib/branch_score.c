/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Branch-score (Kuhner-Felsenstein) distance */

#include <stdlib.h>
#include <math.h>
#include <phast/misc.h>
#include "tree_splits.h"
#include "branch_score.h"

static int split_cmp(const void *pa, const void *pb) {
  const TreeSplit *a = pa, *b = pb;
  return bs_compare(a->mask, b->mask);
}

/* Euclidean distance between two sorted split-length vectors. */
static double split_vector_distance(const TreeSplit *a, int na,
                                    const TreeSplit *b, int nb) {
  double sumsq = 0.0;
  int i = 0, j = 0;
  while (i < na && j < nb) {
    int cmp = bs_compare(a[i].mask, b[j].mask);
    if (cmp == 0) {
      double diff = a[i].blen - b[j].blen;
      sumsq += diff * diff;
      i++;
      j++;
    }
    else if (cmp < 0) {
      sumsq += a[i].blen * a[i].blen;
      i++;
    }
    else {
      sumsq += b[j].blen * b[j].blen;
      j++;
    }
  }
  for (; i < na; i++) sumsq += a[i].blen * a[i].blen;
  for (; j < nb; j++) sumsq += b[j].blen * b[j].blen;
  return sqrt(sumsq);
}

double tr_branch_score(TreeNode *t1, TreeNode *t2) {
  TreeSplitContext *ctx = tr_split_context_new(t1);
  if (tr_split_context_nleaves(ctx) < 2) {
    tr_split_context_free(ctx);
    return 0.0;
  }

  TreeSplitVector *v1 = tr_collect_splits(ctx, t1, TRUE);
  TreeSplitVector *v2 = tr_collect_splits(ctx, t2, TRUE);
  double distance = split_vector_distance(v1->splits, v1->size,
                                          v2->splits, v2->size);
  tr_split_vector_free(v1);
  tr_split_vector_free(v2);
  tr_split_context_free(ctx);
  return distance;
}

double tr_branch_score_pointest(List *trees, TreeNode *ref) {
  int ntrees = lst_size(trees);
  if (ntrees == 0)
    return 0.0;

  TreeSplitContext *ctx = tr_split_context_new(ref);
  if (tr_split_context_nleaves(ctx) < 2) {
    tr_split_context_free(ctx);
    return 0.0;
  }

  int size = 0, capacity = 16;
  TreeSplit *all = smalloc(capacity * sizeof(TreeSplit));
  for (int i = 0; i < ntrees; i++) {
    TreeSplitVector *vec = tr_collect_splits(ctx, lst_get_ptr(trees, i), TRUE);
    while (size + vec->size > capacity) {
      capacity *= 2;
      all = srealloc(all, capacity * sizeof(TreeSplit));
    }
    for (int j = 0; j < vec->size; j++) {
      all[size].mask = bs_clone(vec->splits[j].mask);
      all[size].blen = vec->splits[j].blen;
      size++;
    }
    tr_split_vector_free(vec);
  }
  qsort(all, size, sizeof(TreeSplit), split_cmp);

  TreeSplit *mean = smalloc((size > 0 ? size : 1) * sizeof(TreeSplit));
  int nmean = 0;
  for (int i = 0; i < size; ) {
    int j = i + 1;
    double sum = all[i].blen;
    while (j < size && bs_compare(all[i].mask, all[j].mask) == 0) {
      sum += all[j].blen;
      bs_free(all[j].mask);
      j++;
    }
    mean[nmean].mask = all[i].mask;
    mean[nmean].blen = sum / ntrees;
    nmean++;
    i = j;
  }
  sfree(all);

  TreeSplitVector *refvec = tr_collect_splits(ctx, ref, TRUE);
  double distance = split_vector_distance(mean, nmean,
                                          refvec->splits, refvec->size);
  for (int i = 0; i < nmean; i++)
    bs_free(mean[i].mask);
  sfree(mean);
  tr_split_vector_free(refvec);
  tr_split_context_free(ctx);
  return distance;
}

static double subtree_length(TreeNode *node, TreeNode *parent) {
  double length = 0.0;
  if (parent != NULL)
    length += node->dparent > 0.0 ? node->dparent : 0.0;
  if (node->lchild != NULL && node->lchild != parent)
    length += subtree_length(node->lchild, node);
  if (node->rchild != NULL && node->rchild != parent)
    length += subtree_length(node->rchild, node);
  return length;
}

double tr_tree_length(TreeNode *tree) {
  return subtree_length(tree, NULL);
}
