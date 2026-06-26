/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* prior distributions over trees and branch lengths, for use in
   variational inference */

#ifndef TREEPRIOR_H
#define TREEPRIOR_H

#include <stdio.h>
#include <phast/trees.h>
#include <phast/tree_model.h>
#include <phast/matrix.h>
#include <bitset.h>

struct cvdat;

/* types of priors over trees and branch lengths */
enum tree_prior_type {YULE, GAMMA, NONE};

/* metadata for prior; allows for Yule or Gamma model for trees, and
   orthogonally, for a relaxed local clock model (or not) for absolute
   substitution rates along branches.  The clock model uses a
   lognormal prior (with mean 1) for relative substitution rates on
   each branch. The standard deviation (sigma) of this model is
   estimated from the data.  The prior for sigma may be defined as a
   normal prior on log sigma or an exponential prior on sigma (see
   below).  */
typedef struct {
  enum tree_prior_type type; /* YULE or GAMMA for trees and branch lengths,
                           or NONE to ignore tree prior */
  double gamma_shape;  /* hyperparameter: shape of gamma prior (if GAMMA) */
  double gamma_scale;  /* scale of gamma prior; estimated by empirical
                          Bayes from initial tree */
  unsigned int relclock; /* TRUE or FALSE; whether or not relaxed
                            local clock is active.  All relevant
                            parameter are ignored if FALSE */
  double relclock_sig; /* stdev of log normal prior for relaxed clock
                          (estimated).  Actual value of sigma is
                          obtained by applying softplus() to this raw
                          value */
  double relclock_sig_grad; /* gradient of raw sigma (before
                               softplus), for use in gradient
                               ascent */
  double relclock_lsig_mean; /* hyperparameter for mean of log sigma
                                or -1 to use exponential prior
                                instead */
  double relclock_lsig_sd; /* hyperparameter for sd of log sigma if
                              active */
  double relclock_sig_exp_mean; /* hyperparameter for mean of exp
                                   sigma if active (or -1) */
  Vector *nodetimes; /* absolute time of each node; estimated from
                        data. Stored as raw values to which softplus
                        is applied to obtain actual times */
  Vector *nodetimes_grad; /* gradient of raw nodetimes (before softplus) */
  double tau_beta;  /* scale parameter for scaled softplus
                       parameterization of nodetimes; set from branch
                       lengths on initialization */
  BSHash *bs2idx; /* bitset hash used to index internal nodes by sets of
                     descendant leaves; needed for persistence of
                     nodetimes as different trees are sampled */
  Vector *rates; /* M2 latent relaxed clock: per-branch relative
                    substitution rates, one per non-root node (size
                    nnodes-1), indexed by br2idx slot.  Optimized as
                    nuisance parameters; initialized to 1.0.  bl_eff =
                    rate_b * tau_b enters the Felsenstein likelihood. */
  Vector *rates_grad; /* gradient of rates: the likelihood part
                         (tau_b * dL/d bl_eff, set in gradients.c) plus the
                         lognormal-prior part (set in tp_compute_log_prior) */
  BSHash *br2idx; /* bitset hash mapping each branch (a non-root node's
                     descendant-leaf set, incl. leaf singletons) to its
                     slot in rates; persists rates across sampled trees */
} TreePrior;

/* shape parameter for Gamma prior; use moderately informative value */
#define GAMMA_SHAPE 2

/* constants for sigma, the standard deviation of the lognormal
   distribution for relative clock rates along branches */
/* initialization for sigma */
#define SIG_INIT 0.5
/* floor for sigma in estimation, to keep from collapsing.  Kept small
   (just enough to bound the 1/sigma terms); the old value of 0.4
   pinned the clock far above the data-supported scale (Tier 0). */
#define SIG_FLOOR 0.1
/* hyperparameter for mean of log sigma */
#define LSIG_MEAN log(0.7)
/* hyperparameter for sd of log sigma.  Widened from 0.1 (which was a
   +/-10%-in-log straitjacket pinning sigma ~0.7 regardless of data) so
   the clock can adapt to the data-supported rate variance (Tier 0). */
#define LSIG_SD 0.5
/* hyperparameter for mean of sigma under exponential */
#define SIG_EXP_MEAN 2

TreePrior *tp_new(enum tree_prior_type type, unsigned int relclock);

void tp_free(TreePrior *tp);

double tp_compute_log_prior(TreeModel *mod, struct cvdat *data, Vector *branchgrad);

double tp_prior_noclock(TreeModel *mod, TreePrior *tp, Vector *branchgrad);

void tp_init_nodetimes(TreePrior *tp, TreeModel *mod, List *bs_by_id);

/* M2 latent relaxed clock: rescale each non-root node's dparent from the
   internode time tau_b to the effective branch length bl_eff = rate_b*tau_b
   before the Felsenstein likelihood call.  Saves the original tau into
   tau_saved (size nnodes, indexed by node id); allocates rates/br2idx on
   first call and zeroes rates_grad for this gradient evaluation. */
void tp_rates_pre_likelihood(TreeModel *mod, TreePrior *tp, Vector *tau_saved);

/* M2: after the likelihood call, dL_dt[b] = dL/d bl_eff.  Accumulate the
   likelihood part of rates_grad (tau_b * dL_dt[b]), rewrite dL_dt[b] in
   place to the time gradient (rate_b * dL_dt[b]) for the UPGMA backprop, and
   restore dparent = tau_saved. */
void tp_rates_post_likelihood(TreeModel *mod, TreePrior *tp, Vector *tau_saved,
                              Vector *dL_dt);

void tp_init_gamma_scale(TreePrior *tp, TreeModel *mod);

double tp_treelen(TreeModel *mod);

#endif 
