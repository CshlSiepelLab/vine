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


/* these structs are used to record the choices of neighbors during
   a forward pass of the algorithm, to facilitate backpropagation
   afterward */
typedef struct {
  int u, v, w;           /* indices of the merged leaves/clusters */
  int nk;                /* no. active taxa at this step */
  int branch_idx_u;      /* which slot in dL_dt corresponds to branch u->w */
  int branch_idx_v;      /* which slot in dL_dt corresponds to branch v->w */
  double d_uv;           /* distance between u and v at this step */
  double row_sum_u;      /* sum_m d_{um} over all other active m (includes v) */
  double row_sum_v;      /* sum_m d_{vm} over all other active m (includes u) */
} JoinEvent;

typedef struct neigh_struc {
  int n;                 /* no. original taxa */
  int total_nodes;       /* 2n-2 */
  int nsteps;            /* number of recorded merges (n-2) */
  JoinEvent *steps;      /* length nsteps, in forward order */
  int root_u;               /* index of first final remaining node */
  int root_v;               /* index of second final remaining node */
  int branch_idx_root_u;    /* dL_dt index for branch root_u->root */
  int branch_idx_root_v;    /* dL_dt index for branch root_v->root */
} Neighbors;

void nj_resetQ(Matrix *Q, Matrix *D, Vector *active, Vector *sums, int *u,
	       int *v, int maxidx);

void nj_updateD(Matrix *D, int u, int v, int w, Vector *active, Vector *sums);

TreeNode *nj_infer_tree(Matrix *initD, char **names, Matrix *dt_dD,
                        Neighbors *nb);

TreeNode *nj_fast_infer(Matrix *initD, char **names, Matrix *dt_dD,
                        Neighbors *nb);

NJHeapNode* nj_heap_computeQ(int i, int j, int n, Matrix *D,
                             Vector *sums, int *rev);

double nj_compute_JC_dist(MSA *msa, int i, int j);

Matrix *nj_compute_JC_matr(MSA *msa);

Matrix *nj_tree_to_distances(TreeNode *tree, char **names, int n);

double nj_distance_on_tree(TreeNode *root, TreeNode *n1, TreeNode *n2);

TreeNode *nj_inf(Matrix *D, char **names, Matrix *dt_dD, Neighbors *nb,
                 struct cvdat *covar_data);

void nj_update_seq_to_node_map(TreeNode *tree, char **names, struct cvdat *data);

void nj_update_diam_leaves(Matrix *D, struct cvdat *data);

void nj_repair_zero_br(TreeNode *t);

Neighbors *nj_new_neighbors(int n);

void nj_free_neighbors(Neighbors *nb);

void nj_copy_neighbors(Neighbors *dest, Neighbors *src);

void nj_record_join(Neighbors *nb, int step_idx, int u, int v, int w,
                    Vector *active, Vector *sums, Matrix *D, int branch_idx_u,
                    int branch_idx_v);

void nj_dL_dD_from_neighbors(const Neighbors *nb, Vector *dL_dt,
                             Vector *dL_dD);
#endif
