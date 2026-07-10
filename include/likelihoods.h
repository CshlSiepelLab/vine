/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
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

/* forward declaration (defined in covariance.h) */
#ifndef COV_H
typedef struct cvdat CovarData;
#endif

/* number of free parameters in GTR model */
#define GTR_NPARAMS 6

/* Per-thread derivative accumulators (thread-local) */
typedef struct NJDerivs {
  Vector *branchgrad;      /* [nbranches], mixed across categories */
  double deriv_hky_kappa;  /* only if HKY85 */
  Vector *deriv_gtr;       /* only if REV */
  double deriv_dgamma_alpha; /* only if dgamma_cats > 1 */

  /* below for CRISPR case */
  double deriv_leading_t; /* partial deriv wrt leading branch length */
  double deriv_sil;       /* partial deriv wrt silencing rate */
  unsigned int zero_likl; /* likelihood evaluated to zero */
} NJDerivs;

/* read-only cache of gradient matrices (shared across threads) */
typedef struct NJGradCache {
  Matrix ***grad_mat;          /* [nnodes][ncats] */
  Matrix ***grad_mat_HKY;      /* [nnodes][ncats] or NULL */
  List ***grad_mat_REV;        /* [nnodes][ncats] or NULL */
  Vector *tuplecounts;        /* counts of each unique tuple */
} NJGradCache;

void reset_tree_model(TreeModel *mod, TreeNode *newtree);

double ll_core(TreeModel *mod, CovarData *data, NJDerivs *derivs,
                NJGradCache *gcache, List *range);

int *build_seq_idx(List *leaves, char **names);

int get_seq_idx(char **names, char *name, int n);

void init_gtr_mapping(TreeModel *tm);

double compute_log_likelihood(TreeModel *mod, CovarData *data, Vector *branchgrad);

double ll_parallel(TreeModel *mod, CovarData *data, Vector *branchgrad,
                      int nthreads_requested, NJGradCache *gcache);

#endif
