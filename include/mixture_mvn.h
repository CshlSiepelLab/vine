/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef MIXTURE_MVN_H
#define MIXTURE_MVN_H

#include <multi_mvn.h>
#include <covariance.h>

/* Mixture of embedding distributions. */
typedef struct mixture_MVN {
  int ncomponents;
  multi_MVN **components;
  Vector *logits;
  Vector *weights;
} mixture_MVN;

mixture_MVN *mixmvn_new(int ncomponents, int n, int d, enum mvn_type type);

void mixmvn_free(mixture_MVN *mix);

void mixmvn_init_jitter_from_component(mixture_MVN *mix, int component,
                                  double jitter_sd);

void mixmvn_update_covariance(mixture_MVN *mix, CovarData *data);

void mixmvn_update_weights(mixture_MVN *mix);

int mixmvn_sample_component(mixture_MVN *mix);

double mixmvn_log_dens(mixture_MVN *mix, Vector *x);

void mixmvn_print_embedding(mixture_MVN *mix, char **names, int n, int d,
                            FILE *F);

#endif
