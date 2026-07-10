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

/* Branch-score (Kuhner-Felsenstein) distance between two trees, computed on
 * the unrooted split -> branch-length vectors, including terminal (leaf)
 * edges.  Leaf sets must match. */
double tr_branch_score(TreeNode *t1, TreeNode *t2);

/* Branch-score distance between the posterior-MEAN split-length vector of a
 * collection of trees and a reference tree.  For each split the mean length is
 * the sum of that split's length over all trees containing it, divided by the
 * total number of trees (absent = length 0) -- the L2-optimal point estimate
 * that minimizes expected BSD.  For a single-tree collection this equals
 * tr_branch_score(tree, ref).  Isolates branch-length accuracy from posterior
 * dispersion. */
double tr_branch_score_pointest(List *trees, TreeNode *ref);

/* Total branch length of a tree (sum of all edge lengths; rooting-invariant). */
double tr_tree_length(TreeNode *t);

/* Compute split entropy, topology entropy, and mean branch-length variance
 * for a collection of trees (each element a TreeNode*).
 * H_split:            sum of Bernoulli entropies over non-trivial splits.
 * H_top:              Shannon entropy over distinct topologies.
 * mean_var:           topology-weighted sum of sample variances of log
 *                     branch lengths, summed over all branches.
 * mean_var_per_branch: mean_var / m  (m = number of branches per tree). */
void tr_tree_entropy(List *trees, double *H_split, double *H_top,
                     double *mean_var, double *mean_var_per_branch);

#endif
