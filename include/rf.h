/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* calculation of Robinson Foulds distances */

#ifndef RF_H
#define RF_H

#include <stdio.h>
#include <stdint.h>            /* uint64_t */
#include <phast/lists.h>       /* List */
#include <phast/trees.h>       /* TreeNode */

/* bitset for up to many thousands of leaves */
typedef struct {
  int W;            /* number of 64-bit words */
  uint64_t *w;      /* words */
} BitMask;

/* dynamic array of BitMask* */
typedef struct {
  BitMask **a; int size, cap;
} MaskVec;

double tr_robinson_foulds(TreeNode *t1, TreeNode *t2);

/* Compute split entropy, topology entropy, and mean branch-length variance
 * for a collection of trees (each element a TreeNode*).
 * H_split:            sum of Bernoulli entropies over non-trivial splits.
 * H_top:              Shannon entropy over distinct topologies.
 * mean_var:           topology-weighted sum of sample variances of log
 *                     branch lengths, summed over all branches.
 * mean_var_per_branch: mean_var / m  (m = number of branches per tree). */
void tr_tree_entropy(List *trees, double *H_split, double *H_top,
                     double *mean_var, double *mean_var_per_branch);

/* lexicographic comparison of two canonical splits (see BitMask above) */
int tr_bitmask_cmp(const BitMask *a, const BitMask *b);

/* free a BitMask returned by tr_collect_split_counts */
void tr_bitmask_free(BitMask *m);

/* (canonical split, occurrence count) pair, as returned by
 * tr_collect_split_counts. */
typedef struct { BitMask *mask; int count; } SplitCount;

/* Collect unique non-trivial splits and their occurrence counts across a
 * set of trees.  Returns a newly allocated array sorted by mask, with
 * *nsc_out set to its length.  *nleaves_out (if non-NULL) is set to the
 * number of leaves.  Returns NULL (with *nsc_out == 0) if trees is empty
 * or has fewer than 3 leaves.  Caller frees each element's mask via
 * tr_bitmask_free, then frees the returned array itself. */
SplitCount *tr_collect_split_counts(List *trees, int *nsc_out, int *nleaves_out);

#endif
