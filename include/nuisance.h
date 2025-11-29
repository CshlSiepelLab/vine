/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#ifndef NUIS_H
#define NUIS_H

#include <stdio.h>
#include <limits.h>
#include <phast/tree_model.h>
#include <covariance.h>


int nj_get_num_nuisance_params(TreeModel *mod, CovarData *data);

char *nj_get_nuisance_param_name(TreeModel *mod, CovarData *data, int idx);

void nj_update_nuis_grad(TreeModel *mod, CovarData *data, Vector *nuis_grad);

void nj_save_nuis_params(Vector *stored_vals, TreeModel *mod, CovarData *data);

void nj_update_nuis_params(Vector *stored_vals, TreeModel *mod, CovarData *data);

void nj_nuis_param_pluseq(TreeModel *mod, CovarData *data, int idx, double inc);

double nj_nuis_param_get(TreeModel *mod, CovarData *data, int idx);

#endif
