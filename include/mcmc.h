/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* MCMC refinement of variational samples */

#ifndef MCMC_H
#define MCMC_H

#include <stdio.h>
#include <phast/tree_model.h>
#include <mixture_mvn.h>
#include <nj.h>

#define BURNIN_ITERS 400
#define TUNING_INTERVAL 10
#define TARGET_ACCEPT_RATE 0.3
#define MIN_S 0.001

List *var_sample_mcmc(int nsamples, int thin, mixture_MVN *mixmvn,
                         CovarData *data, TreeModel *mod, FILE *logf,
                         unsigned int silent);

#endif
