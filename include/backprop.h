/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef BCKP_H
#define BCKP_H

#include <stdio.h>
#include <limits.h>
#include <backprop.h>
#include <sparse_matrix.h>


int nj_i_j_to_dist(int i, int j, int n);

void nj_dist_to_i_j(int pwidx, int *i, int *j, int n);

void nj_backprop_sparse(SparseMatrix *Jk, SparseMatrix *Jnext, int n, int f, int g, int u,
                        Vector *active);

void nj_backprop_init_sparse(SparseMatrix *Jk, int n);

void nj_backprop_set_dt_dD_sparse(SparseMatrix *Jk, Matrix *dt_dD, int n, int f, int g,
                                  int branch_idx_f, int branch_idx_g, Vector *active);


#endif
