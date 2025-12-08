/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef UPGMA_H
#define UPGMA_H

#include <stdio.h>
#include <phast/matrix.h>
#include <phast/trees.h>
#include <phast/tree_model.h>
#include <phast/misc.h>

/* for use with min-heap in fast upgma algorithm */
typedef struct {
  int i, j;
  double val;  // raw distance D(i, j)
} UPGMAHeapNode;

void upgma_find_min(Matrix *D, Vector *active, int *u, int *v);

void upgma_updateD(Matrix *D, int u, int v, int w, Vector *active,
                   Vector *sizes, Vector *heights);

TreeNode* upgma_infer_tree(Matrix *initD, char **names, Matrix *dt_dD);

void upgma_set_dt_dD(TreeNode *tree, Matrix* dt_dD);

UPGMAHeapNode* upgma_heap_node(int i, int j, Matrix *D);

TreeNode* upgma_fast_infer(Matrix *initD, char **names, Matrix *dt_dD);

void upgma_dL_dD_from_tree(TreeNode *tree, Vector *dL_dt, Vector *dL_dD);

#endif
