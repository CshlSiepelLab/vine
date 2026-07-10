/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef TREE_SPLITS_H
#define TREE_SPLITS_H

#include <phast/lists.h>
#include <phast/trees.h>
#include <bitset.h>

typedef struct tr_split_context TreeSplitContext;

typedef struct {
  BSet *mask;
  double blen;
} TreeSplit;

typedef struct {
  TreeSplit *splits;
  int size;
} TreeSplitVector;

typedef struct {
  BSet *mask;
  int count;
} SplitCount;

/* Establish the canonical sorted leaf order used for split masks. */
TreeSplitContext *tr_split_context_new(TreeNode *tree);
void tr_split_context_free(TreeSplitContext *ctx);
int tr_split_context_nleaves(const TreeSplitContext *ctx);

/* Collect a sorted vector of unique canonical splits.  Duplicate masks, such
 * as the two halves of an edge adjacent to a rooted Newick root, are collapsed
 * by summing their branch lengths. */
TreeSplitVector *tr_collect_splits(TreeSplitContext *ctx, TreeNode *tree,
                                   unsigned int include_terminal);
void tr_split_vector_free(TreeSplitVector *vec);

/* Count unique non-terminal splits across a tree sample using ctx's leaf
 * order.  The returned array is sorted by mask. */
SplitCount *tr_collect_split_counts(TreeSplitContext *ctx, List *trees,
                                    int *nsc_out);
void tr_split_counts_free(SplitCount *counts, int ncounts);

#endif
