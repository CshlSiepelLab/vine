/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* KL divergence metrics between two posterior tree samples */

#ifndef KL_H
#define KL_H

#include <phast/lists.h>       /* List */

/* Mean per-split Bernoulli KL divergence KL(p_est || p_ref), where p_est(s)
 * and p_ref(s) are the posterior inclusion probabilities of non-trivial
 * clade split s under the estimate and reference tree samples,
 * respectively.  Averaged over the union of splits observed in either
 * sample (Laplace-smoothed so no split has probability exactly 0 or 1).
 * trees_est and trees_ref must be over the same set of leaf names. */
void tr_split_kl(List *trees_est, List *trees_ref, double *mean_kl);

/* Mean Gaussian KL divergence between the estimate and reference samples'
 * embedded pairwise-distance distributions.  Each tree is independently
 * embedded into a dim-dimensional Euclidean space via classical
 * multidimensional scaling on its cophenetic distances (the same
 * closed-form procedure vine itself uses to initialize its variational
 * posterior; see estimate_mmvn_from_distances).  Pairwise Euclidean
 * distances between the embedded leaves are rotation/reflection/
 * translation-invariant, so no alignment across independently-embedded
 * trees is needed.  For each leaf pair, a Gaussian(mean, sd) is fit to
 * its embedded distance across the sample, and the KL divergences are
 * averaged over all pairs.  If dim <= 0, a default dimensionality is
 * chosen following vine's own default formula (based on the number of
 * leaves).  trees_est and trees_ref must be over the same set of leaf
 * names. */
void tr_embed_kl(List *trees_est, List *trees_ref, int dim, double *mean_kl);

#endif
