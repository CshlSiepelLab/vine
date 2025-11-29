/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef LIK_H
#define LIK_H

#include <stdio.h>
#include <limits.h>
#include <phast/matrix.h>
#include <phast/msa.h>
#include <phast/trees.h>
#include <phast/tree_model.h>
#include <mvn.h>
#include <multi_mvn.h>
#include <sparse_matrix.h>
#include <crispr.h>
#include <phast/misc.h>
#include <radial_flow.h>
#include <planar_flow.h>
#include <tree_prior.h>
#include <migration.h>

/* number of free parameters in GTR model */
#define GTR_NPARAMS 6

void nj_reset_tree_model(TreeModel *mod, TreeNode *newtree);

double nj_compute_log_likelihood(TreeModel *mod, CovarData *data, Vector *branchgrad);

int *nj_build_seq_idx(List *leaves, char **names);

int nj_get_seq_idx(char **names, char *name, int n);

void nj_init_gtr_mapping(TreeModel *tm);

#endif
