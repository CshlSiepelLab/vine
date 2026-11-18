/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Local posterior approximation around a point-estimate tree
   (--posterior).  Topological uncertainty is represented by an
   independent choice among the three nearest-neighbor-interchange
   (NNI) configurations at each internal branch, with probabilities
   proportional to the likelihood of each configuration; branch-length
   uncertainty is represented by a diagonal Laplace approximation in
   branch-length space about the maximum-likelihood branch lengths of
   the point-estimate topology. */

#ifndef LOCAL_POST_H
#define LOCAL_POST_H

#include <phast/trees.h>
#include <phast/tree_model.h>
#include <phast/lists.h>
#include <covariance.h>

/* maximum number of diagonal Newton iterations for branch lengths */
#define LP_MAX_NEWTON 6

/* stop Newton iterations when the log likelihood improves by less
   than this amount */
#define LP_NEWTON_TOL 0.01

/* relative and minimum absolute step for finite-difference curvature */
#define LP_REL_STEP 0.1
#define LP_MIN_STEP 1e-4

List *lp_sample(int nsamples, TreeNode *pointtree, TreeModel *mod,
                CovarData *data, char **names, unsigned int silent);

#endif
