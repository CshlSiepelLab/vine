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
#include <phast/matrix.h>      /* Matrix */
#include <phast/trees.h>       /* TreeNode */

double tr_robinson_foulds(TreeNode *t1, TreeNode *t2);

/* Compute and write an upper-triangular pairwise RF distance matrix. */
Matrix *tr_robinson_foulds_matrix(List *trees);
void tr_write_robinson_foulds_matrix(Matrix *matrix, FILE *F);

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
