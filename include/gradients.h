/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef GRAD_H
#define GRAD_H

#include <stdio.h>
#include <limits.h>
#include <phast/matrix.h>
#include <phast/tree_model.h>
#include <nj.h>
#include <mvn.h>
#include <multi_mvn.h>

/* for numerical derivatives */
#define DERIV_EPS 1e-5

/* constants used for regularization of variance parameters; can be
   altered multiplicatively using --var-reg option */
#define PENALTY_LOGLAMBDA_CONST 5
#define PENALTY_LOGLAMBDA_DIAG 5
#define PENALTY_LOGLAMBDA_LOWR 0.5


double nj_compute_model_grad(TreeModel *mod, multi_MVN *mmvn, 
                             Vector *points, Vector *points_std,
                             Vector *grad, CovarData *data,
                             double *nf_logdet, double *migll);

double nj_compute_model_grad_check(TreeModel *mod, multi_MVN *mmvn, 
                                   Vector *points, Vector *points_std,
                                   Vector *grad, CovarData *data);


void nj_rescale_grad(Vector *grad, Vector *rsgrad, multi_MVN *mmvn,
                     CovarData *data);

void nj_compute_variance_penalty(Vector *grad, multi_MVN *mmvn, CovarData *data);

double nj_dL_dx_dumb(Vector *x, Vector *dL_dx, TreeModel *mod, 
                     CovarData *data);

double nj_dL_dt_num(Vector *dL_dt, TreeModel *mod, CovarData *data);

void nj_dt_dD_num(Matrix *dt_dD, Matrix *D, TreeModel *mod, CovarData *data);

double nj_dL_dx_smartest(Vector *x, Vector *dL_dx, TreeModel *mod, 
                         CovarData *data, double *nf_logdet, double *migll);


#endif
