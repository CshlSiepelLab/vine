/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef NJ_H
#define NJ_H

#include <stdio.h>
#include <limits.h>
#include <phast/matrix.h>
#include <phast/tree_model.h>
#include <covariance.h>

/* for use with min-heap in fast nj algorithm */
typedef struct NJHeapData {
  double val;
  int i, j; 
  int rev_i, rev_j; // for lazy validation
} NJHeapNode;

void nj_resetQ(Matrix *Q, Matrix *D, Vector *active, Vector *sums, int *u,
	       int *v, int maxidx);

void nj_updateD(Matrix *D, int u, int v, int w, Vector *active, Vector *sums);

TreeNode* nj_infer_tree(Matrix *initD, char **names, Matrix *dt_dD);

TreeNode* nj_fast_infer(Matrix *initD, char **names, Matrix *dt_dD);

NJHeapNode* nj_heap_computeQ(int i, int j, int n, Matrix *D,
                             Vector *sums, int *rev);

double nj_compute_JC_dist(MSA *msa, int i, int j);

Matrix *nj_compute_JC_matr(MSA *msa);

Matrix *nj_tree_to_distances(TreeNode *tree, char **names, int n);

double nj_distance_on_tree(TreeNode *root, TreeNode *n1, TreeNode *n2);

TreeNode *nj_inf(Matrix *D, char **names, Matrix *dt_dD,
                 CovarData *covar_data);

void nj_update_seq_to_node_map(TreeNode *tree, char **names, CovarData *data);

void nj_update_diam_leaves(Matrix *D, CovarData *data);

void nj_repair_zero_br(TreeNode *t);

#endif
