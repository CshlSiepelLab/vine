/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* handling of crispr mutation models for phylogeny reconstruction */
/* reads tab-delimited mutation data, calculates substitution probabilities, custom pruning algorithm for better efficiency */

#ifndef CPR_H
#define CPR_H

#include <stdio.h>
#include <phast/tree_model.h>
#include <phast/vector.h>

/* forward declarations to avoid circular includes */
struct NJDerivs;
struct mgtab; /* forward declaration */

/* initialization and bounds for silencing rate */
#define CPR_SIL_RATE_INIT 0.4
#define CPR_SIL_RATE_MAX 100.0

/* branch-length floor for irreversible model; kept small to avoid
   penalizing branches that are optimally zero */
#define CPR_T_FLOOR 1.0e-8

typedef struct {
  int nsites;
  int ncells;
  int nstates;
  List *sitenames;
  List *cellnames;
  List *cellmuts;
  int *sitewise_nstates;
  List **dupnames; /* element i is list of names of cells with same
                    * genotype as cell i, or NULL if no duplicates */
} CrisprMutTable;

/* model for mutations along branches of a phylogenetic tree; allows
   for either a single model for all sites (like TiDeTree) or a
   separate model for each site (like LAML) */
enum crispr_model_type {GLOBAL, SITEWISE};
enum crispr_mutrates_type {UNIF, EMPIRICAL};
typedef struct {
  enum crispr_model_type model_type;
  int nsites;
  int ncells;
  TreeModel *mod; /* encapsulates tree with branch lengths 
                     and some auxiliary data */
  CrisprMutTable *mut;
  int nstates;
  List *Pt;
  enum crispr_mutrates_type mutrates_type;
  Vector *mutrates;  /* global mutation rates */
  List *sitewise_mutrates; /* one vector per site */
  double sil_rate;  /* rate of silencing */
  double deriv_sil; /* latest partial deriv of log likelihood wrt
                       sil_rate */
  double leading_t; /* length of leading branch to root */
  double deriv_leading_t; /* latest partial deriv of leading branch
                             len */
  unsigned int zero_likl; /* likelihood evaluated to zero; typically
                             indicates degenerate tree */
  unsigned int mig_warmup; /* if nonzero, skip migration likelihood and
                              zero migration gradients during warmup phase */
  int nthreads; /* number of threads to use in
                   parallel likelihood calculations */
  struct cpr_anc_state_sets *ancsets; /* per-model ancestral-state-set
                                         cache, lazily allocated by
                                         cpr_compute_log_likelihood and
                                         released by cpr_free_model;
                                         opaque here, defined below */
} CrisprMutModel;

/* auxiliary data used to keep track of restricted ancestral state
   possibilities in likelihood calculation; allows for greatly
   accelerated algorithm */
typedef struct cpr_anc_state_sets {
  int nnodes; /* total number of nodes in tree; root is included but
                 will be ignored */
  int NORESTRICT; /* code indicating no restrictions on state */
  List *unr_lists; /* element i is list containing all integers from 0
                      to i inclusive (cached) */
  List *restr_lists; /* element i is list containing 0 and i
                        (cached) */
  List *sil_lists; /* element i is list consisting only of node i, for i > 1 */
} CrisprAncestralStateSets;

CrisprMutTable *cpr_new_table();

CrisprMutTable *cpr_read_table(FILE *F);

void cpr_free_table(CrisprMutTable *M);

void cpr_print_table(CrisprMutTable *M, FILE *F);

int cpr_get_mut(CrisprMutTable *M, int cell, int site);

void cpr_set_mut(CrisprMutTable *M, int cell, int site, int val);

void cpr_renumber_states(CrisprMutTable *M);

void cpr_deduplicate(CrisprMutTable *M, struct mgtab *mg);

void cpr_add_dup_leaves(TreeNode *tree, CrisprMutTable *M);

void cpr_expand_tables_for_dups(CrisprMutTable *M, struct mgtab *mg);

double cpr_compute_log_likelihood(CrisprMutModel *cprmod,
                                  Vector *branchgrad);

double cpr_ll_core(CrisprMutModel *cprmod, struct NJDerivs *derivs,
                   int *nodetypes, List *range);

double cpr_ll_parallel(CrisprMutModel *cprmod, Vector *branchgrad,
                       int nthreads_requested);

Matrix *cpr_compute_dist(CrisprMutTable *M);

Matrix *cpr_compute_dist_parsimony(CrisprMutTable *M);

Matrix *cpr_compute_dist_hamming(CrisprMutTable *M);

double cpr_compute_pw_dist(CrisprMutTable *M, int i, int j);

double cpr_compute_pw_dist_parsimony(CrisprMutTable *M, int i, int j);

double cpr_compute_pw_dist_nopriv(CrisprMutTable *M, int i, int j,
                                   int **state_counts);

void cpr_set_branch_matrix(MarkovMatrix *P, double t, double silent_rate,
                           Vector *mutrates);

void cpr_branch_grad(Matrix *grad, double t, double silent_rate,
                     Vector *eqfreqs);

void cpr_silent_rate_grad(Matrix *grad, double t, double silent_rate,
                          Vector *mutrates);

Vector *cpr_estim_mutrates(CrisprMutTable *M,
                            enum crispr_mutrates_type type);

List *cpr_estim_sitewise_mutrates(CrisprMutTable *M,
                                   enum crispr_mutrates_type type);

void cpr_build_seq_idx(TreeModel *mod, CrisprMutTable *M);

CrisprAncestralStateSets *cpr_new_state_sets(int nnodes);

void cpr_state_sets_resize(CrisprAncestralStateSets *sets, int newnnodes);

List *cpr_get_state_set(CrisprAncestralStateSets *set, int *nodetypes,
                        TreeNode *n, int nstates);

void cpr_prepopulate_state_sets(CrisprAncestralStateSets *sets,
                                int max_nstates);

void cpr_free_state_sets(CrisprAncestralStateSets *sets);

CrisprMutTable *cpr_new_sitewise_table(CrisprMutTable *origM);

CrisprMutModel *cpr_new_model(CrisprMutTable *M, TreeModel *mod,
                              enum crispr_model_type modtype,
                              enum crispr_mutrates_type eqtype);

void cpr_prep_model(CrisprMutModel *cprmod);

void cpr_print_model(CrisprMutModel *cprmod, FILE *F);

void cpr_free_model(CrisprMutModel *cprmod);

void cpr_check_dedup_tables(CrisprMutTable *M, struct mgtab *mg,
                            const char *stage);

#endif
