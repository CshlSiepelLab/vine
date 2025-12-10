/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef VAR_H
#define VAR_H

#include <stdio.h>
#include <limits.h>
#include <phast/tree_model.h>
#include <nj.h>
#include <mvn.h>
#include <multi_mvn.h>

/* tuning parameters for Adam algorithm.  These will be kept at the
   default values.  The learning rate (called alpha) will be passed in
   as a parameter */
#define ADAM_BETA1 0.9
//#define ADAM_BETA2 0.999
#define ADAM_BETA2 0.9
#define ADAM_EPS 1e-8

/* starting number of columns of alignment to subsample in early stages of
   algorithms */
#define NSUBSAMPLES 256


void nj_variational_inf(TreeModel *mod, multi_MVN *mmvn,
                        int nminibatch, double learnrate, int nbatches_conv,
                        int min_nbatches, CovarData *data, FILE *logf);

double nj_elbo_montecarlo(TreeModel *mod, multi_MVN *mmvn, CovarData *data,
                          int nminibatch, Vector *avegrad,
                          Vector *ave_nuis_grad, double *ave_lprior,
                          double *avemigll);

List *nj_var_sample(int nsamples, multi_MVN *mmvn, CovarData *data,
                    char** names, Vector *logdens);

TreeNode *nj_mean(Vector *mu, char **names, CovarData *data);

void nj_sample_points(multi_MVN *mmvn, Vector *points,
                      Vector *points_std);

void nj_apply_normalizing_flows(Vector *points_y, Vector *points_x,
                                CovarData *data, double *logdet);

void nj_set_kld_grad_LOWR(Vector *kldgrad, multi_MVN *mmvn);

void nj_set_entropy_grad_LOWR(Vector *entgrad, multi_MVN *mmvn);

#endif
