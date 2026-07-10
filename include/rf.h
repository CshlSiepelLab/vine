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
#include <phast/lists.h>       /* List */
#include <phast/trees.h>       /* TreeNode */
#include <bitset.h>            /* BSet */

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

/* (canonical split, occurrence count) pair, as returned by
 * tr_collect_split_counts. */
typedef struct { BSet *mask; int count; } SplitCount;

/* Collect unique non-trivial splits and their occurrence counts across a
 * set of trees.  Returns a newly allocated array sorted by mask, with
 * *nsc_out set to its length.  *nleaves_out (if non-NULL) is set to the
 * number of leaves.  Returns NULL (with *nsc_out == 0) if trees is empty
 * or has fewer than 3 leaves.  Caller frees each element's mask via
 * bs_free, then frees the returned array itself. */
SplitCount *tr_collect_split_counts(List *trees, int *nsc_out, int *nleaves_out);

#endif
