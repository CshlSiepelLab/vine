/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <stdlib.h>
#include <phast/misc.h>
#include <phast/stringsplus.h>
#include <phast/hashtable.h>
#include "tree_splits.h"

struct tr_split_context {
  int nleaves;
  List *names;
  Hashtable *name_to_idx;
};

static int split_cmp(const void *pa, const void *pb) {
  const TreeSplit *a = pa, *b = pb;
  return bs_compare(a->mask, b->mask);
}

static int mask_ptr_cmp(const void *pa, const void *pb) {
  const BSet *a = *(BSet * const *)pa;
  const BSet *b = *(BSet * const *)pb;
  return bs_compare(a, b);
}

static BSet *canonical_split(const BSet *mask, unsigned int include_terminal) {
  int size = bs_popcount(mask);
  int other = mask->nbits - size;
  if (size == 0 || other == 0)
    return NULL;
  if (!include_terminal && (size == 1 || other == 1))
    return NULL;

  if (size < other || (size == other && bs_get_bit(mask, 0)))
    return bs_clone(mask);

  BSet *canonical = bs_clone(mask);
  bs_flip(canonical);
  return canonical;
}

static BSet *collect_recurse(TreeSplitContext *ctx, TreeNode *node,
                             TreeNode *parent, unsigned int include_terminal,
                             TreeSplit *splits, int *nsplits) {
  if (node->lchild == NULL && node->rchild == NULL) {
    void *value = hsh_get(ctx->name_to_idx, node->name);
    if (value == (void *)-1)
      die("ERROR: leaf '%s' is not in the expected leaf set.\n", node->name);
    BSet *mask = bs_new(ctx->nleaves);
    bs_set_bit(mask, ptr_to_int(value));
    return mask;
  }

  BSet *subtree = bs_new(ctx->nleaves);
  TreeNode *children[2] = {node->lchild, node->rchild};
  for (int i = 0; i < 2; i++) {
    TreeNode *child = children[i];
    if (child == NULL || child == parent)
      continue;
    BSet *child_mask = collect_recurse(ctx, child, node, include_terminal,
                                       splits, nsplits);
    BSet *canonical = canonical_split(child_mask, include_terminal);
    if (canonical != NULL) {
      splits[*nsplits].mask = canonical;
      splits[*nsplits].blen = child->dparent > 0.0 ? child->dparent : 0.0;
      (*nsplits)++;
    }
    for (int w = 0; w < subtree->nwords; w++)
      subtree->w[w] |= child_mask->w[w];
    bs_free(child_mask);
  }
  return subtree;
}

TreeSplitContext *tr_split_context_new(TreeNode *tree) {
  TreeSplitContext *ctx = smalloc(sizeof(TreeSplitContext));
  ctx->names = tr_leaf_names(tree);
  lst_qsort_str(ctx->names, ASCENDING);
  ctx->nleaves = lst_size(ctx->names);
  ctx->name_to_idx = hsh_new(2 * (ctx->nleaves > 0 ? ctx->nleaves : 2));
  for (int i = 0; i < ctx->nleaves; i++) {
    String *name = lst_get_ptr(ctx->names, i);
    hsh_put_int(ctx->name_to_idx, name->chars, i);
  }
  return ctx;
}

void tr_split_context_free(TreeSplitContext *ctx) {
  if (ctx == NULL)
    return;
  hsh_free(ctx->name_to_idx);
  lst_free_strings(ctx->names);
  lst_free(ctx->names);
  sfree(ctx);
}

int tr_split_context_nleaves(const TreeSplitContext *ctx) {
  return ctx->nleaves;
}

TreeSplitVector *tr_collect_splits(TreeSplitContext *ctx, TreeNode *tree,
                                   unsigned int include_terminal) {
  int capacity = 2 * ctx->nleaves + 2;
  TreeSplit *raw = smalloc((capacity > 0 ? capacity : 1) * sizeof(TreeSplit));
  int nraw = 0;
  BSet *root = collect_recurse(ctx, tree, NULL, include_terminal, raw, &nraw);
  if (bs_popcount(root) != ctx->nleaves)
    die("ERROR: tree does not have the expected leaf set.\n");
  bs_free(root);

  qsort(raw, nraw, sizeof(TreeSplit), split_cmp);
  int nout = 0;
  for (int i = 0; i < nraw; ) {
    int j = i + 1;
    double length = raw[i].blen;
    while (j < nraw && bs_compare(raw[i].mask, raw[j].mask) == 0) {
      length += raw[j].blen;
      bs_free(raw[j].mask);
      j++;
    }
    raw[nout].mask = raw[i].mask;
    raw[nout].blen = length;
    nout++;
    i = j;
  }

  TreeSplitVector *vec = smalloc(sizeof(TreeSplitVector));
  vec->splits = raw;
  vec->size = nout;
  return vec;
}

void tr_split_vector_free(TreeSplitVector *vec) {
  if (vec == NULL)
    return;
  for (int i = 0; i < vec->size; i++)
    bs_free(vec->splits[i].mask);
  sfree(vec->splits);
  sfree(vec);
}

SplitCount *tr_collect_split_counts(TreeSplitContext *ctx, List *trees,
                                    int *nsc_out) {
  int ntrees = lst_size(trees), size = 0, capacity = 16;
  BSet **all = smalloc(capacity * sizeof(BSet *));

  for (int i = 0; i < ntrees; i++) {
    TreeSplitVector *vec = tr_collect_splits(ctx, lst_get_ptr(trees, i), FALSE);
    while (size + vec->size > capacity) {
      capacity *= 2;
      all = srealloc(all, capacity * sizeof(BSet *));
    }
    for (int j = 0; j < vec->size; j++)
      all[size++] = bs_clone(vec->splits[j].mask);
    tr_split_vector_free(vec);
  }

  qsort(all, size, sizeof(BSet *), mask_ptr_cmp);
  SplitCount *counts = smalloc((size > 0 ? size : 1) * sizeof(SplitCount));
  int ncounts = 0;
  for (int i = 0; i < size; ) {
    int j = i + 1;
    while (j < size && bs_compare(all[i], all[j]) == 0)
      j++;
    counts[ncounts].mask = bs_clone(all[i]);
    counts[ncounts].count = j - i;
    ncounts++;
    i = j;
  }
  for (int i = 0; i < size; i++)
    bs_free(all[i]);
  sfree(all);
  *nsc_out = ncounts;
  return counts;
}

void tr_split_counts_free(SplitCount *counts, int ncounts) {
  for (int i = 0; i < ncounts; i++)
    bs_free(counts[i].mask);
  sfree(counts);
}
