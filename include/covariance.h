/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef COV_H
#define COV_H

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
#include <taylor.h>

/* use this as a floor for variance parameters.  Avoids drift to ever
   smaller values */
#define VARFLOOR 1.0e-3

/* initialization of lambda, which is scale factor for covariance
   matrix in DIST and CONST parameterizations */
#define LAMBDA_INIT 1.0e-2

/* types of parameterization for covariance matrix: constant (and
   diagonal), diagonal with free variances, proportional to Laplacian
   pseudoinverse based on pairwise distances, or low-rank
   approximation to full matrix */
enum covar_type {CONST, DIAG, DIST, LOWR};
  
/* auxiliary data for parameterization of covariance matrix; also
   contains other metadata needed for optimization */
typedef struct cvdat {
  enum covar_type type; /* type of parameterization */
  int nseqs; /* number of taxa in tree */
  int dim; /* dimension of point embedding */
  enum mvn_type mvn_type;
  Vector *params; /* vector of free parameters */
  double lambda;  /* scale parameter for covariance matrix 
                     (DIST or CONST cases) */
  double pointscale; /* scale factor for geometry */
  unsigned int natural_grad; /* whether to rescale for natural
                                gradients during optimization */
  double kld_upweight; /* optional upweighting factor for KLD in ELBO */
  Matrix *dist;   /* distance matrix on which covariance is based */
  int lowrank;  /* dimension of low-rank approximation if LOWR or -1
                   otherwise */
  Matrix *Lapl_pinv;  /* Laplacian pseudoinverse (DIST) */
  Vector *Lapl_pinv_evals; /* eigendecomposition of Lapl_pinv (DIST) */
  Matrix *Lapl_pinv_evecs;
  Matrix *R; /* used for LOWR; has dimension lowrank x nseqs */
  double var_reg; /* multiplier for variance regularizer */
  double var_pen; /* the current value of the variance penalty */
  unsigned int hyperbolic; /* whether or not hyperbolic geometry is used */
  double negcurvature; /* for hyperbolic case */
  MSA *msa;            /* multiple alignment under analysis if available */
  CrisprMutModel *crispr_mod; /* model for CRISPR mutation if needed */
  unsigned int ultrametric;   /* whether or not tree is ultrametric */
  MigTable *migtable; /* migration table if needed */
  double hky_kappa; /* for use in estimating kappa as a nuisance
                       parameter in HKY case */
  double deriv_hky_kappa;
  Vector *gtr_params; /* for use in estimating GTR parameters as
                         nuisance parameters in REV case */
  Vector *deriv_gtr;
  char **names;
  unsigned int no_zero_br; /* force all branches to be nonzero;
                              sometimes needed with CRISPR model */
  RadialFlow *rf; /* optional flow layers (NULL if none) */
  PlanarFlow *pf;
  TreePrior *treeprior; /* optional prior for tree (NULL if none) */
  unsigned int subsample; /* whether or not to subsample sites in likelihood calculation */
  int subsampsize; /* size of subsample (number of sites) */
  int reuse_subsamp; /* whether or not to reuse last subsample */
  int tree_diam_leaf1; /* store the ids of two leaves along a diameter
                          of the last reconstructed tree; for use in
                          rerooting tree in case of prior */
  int tree_diam_leaf2;
  int *seq_to_node_map; /* mapping from sequence indices to leaf node
                           ids, used with leaf1 and leaf2 */
  struct taylor_data *taylor;   /* auxiliary data for Taylor approximation to
                                   ELBO (or NULL) */
} CovarData;

void nj_update_covariance(multi_MVN *mmvn, CovarData *data);

CovarData *nj_new_covar_data(enum covar_type covar_param, Matrix *dist, int dim,
                             MSA *msa, CrisprMutModel *crispr_mod, char **names,
                             unsigned int natural_grad, double kld_upweight,
                             int rank, double var_reg, unsigned int hyperbolic,
                             double negcurvature, unsigned int ultrametric,
                             unsigned int radial_flow, unsigned int planar_flow,
                             TreePrior *treeprior, MigTable *mtable,
                             unsigned int use_taylor);

void nj_dump_covar_data(CovarData *data, FILE *F);

void nj_laplacian_pinv(CovarData *data);

unsigned int nj_var_at_floor(multi_MVN *mmvn, CovarData *data);

#endif
