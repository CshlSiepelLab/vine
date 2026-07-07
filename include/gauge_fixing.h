/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef GAUGE_FIXING_H
#define GAUGE_FIXING_H

#include <phast/vector.h>
#include <covariance.h>
#include <mixture_mvn.h>

Vector *gauge_fixing_new_euclidean_reference(mixture_MVN *mixmvn,
                                             CovarData *data);

void gauge_fixing_apply_euclidean(mixture_MVN *mixmvn, CovarData *data,
                                  Vector *reference_mu, Vector **mu_moment);

#endif
