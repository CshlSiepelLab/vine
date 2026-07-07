/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
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
#include <mixture_mvn.h>
#include <adam_scheduler.h>

/* starting number of columns of alignment to subsample in early stages of
   algorithms */
#define NSUBSAMPLES 256

void variational_inf(TreeModel *mod, mixture_MVN *mixmvn, int nminibatch,
                        double learnrate, int nbatches_conv, int min_nbatches,
                        CovarData *data, FILE *logf, unsigned int silent,
                        unsigned int log_all_params);

double elbo_montecarlo(TreeModel *mod, mixture_MVN *mixmvn, int component,
                          CovarData *data,
                          int nminibatch, Vector *model_param_grad,
                          Vector *nuis_param_grad, double *ave_lprior,
                          double *avemigll, double *kld,
                          Vector **sigma_kld_param_grad,
                          Vector **mu_kld_param_grad,
                          Vector *weight_kld_param_grad);

List *var_sample(int nsamples, mixture_MVN *mixmvn, CovarData *data,
                    char** names, Vector *logdens);

TreeNode *infer_distance_tree(Matrix *D, char **names, Matrix *dt_dD,
                              Neighbors *nb, CovarData *data);

TreeNode *mean_tree(Vector *mu, char **names, CovarData *data);

void sample_points(multi_MVN *mmvn, Vector *points,
                      Vector *points_std, int sample_idx);

void apply_normalizing_flows(Vector *points_y, Vector *points_x,
                                CovarData *data, int component,
                                double *logdet);

void set_kld_sigma_grad_LOWR(Vector *sigma_kldgrad, multi_MVN *mmvn);

void set_entropy_sigma_grad_LOWR(Vector *sigma_entgrad, multi_MVN *mmvn);

#endif
