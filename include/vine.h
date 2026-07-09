/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef VINE_H
#define VINE_H

#include <math.h>

/* default embedding dimensionality is a linear function of log number of taxa */
#define DEFAULT_DIM_INTERCEPT 3.25
#define DEFAULT_DIM_SLOPE 0.92

static inline int vine_default_dim(int ntaxa) {
  return (int) round(DEFAULT_DIM_INTERCEPT + DEFAULT_DIM_SLOPE * log((double)ntaxa));
}

#endif /* VINE_H */
