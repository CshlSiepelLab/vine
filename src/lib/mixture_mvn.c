/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <phast/misc.h>
#include <covariance.h>
#include <mixture_mvn.h>


/* Allocate memory and initialize each component mvn. */
mixture_MVN *mixmvn_new(int ncomponents, int n, int d, enum mvn_type type) {
  int k;
  mixture_MVN *mix;

  mix = smalloc(sizeof(mixture_MVN));
  mix->ncomponents = ncomponents;
  mix->components = smalloc(ncomponents * sizeof(multi_MVN*));
  mix->logits = vec_new(ncomponents);
  mix->weights = vec_new(ncomponents);
  vec_zero(mix->logits);

  for (k = 0; k < ncomponents; k++)
    mix->components[k] = mmvn_new(n, d, type);

  mixmvn_update_weights(mix);
  return mix;
}

/* Wrapper to free each component mvn. */
void mixmvn_free(mixture_MVN *mix) {
  int k;

  for (k = 0; k < mix->ncomponents; k++)
    if (mix->components[k] != NULL)
      mmvn_free(mix->components[k]);

  free(mix->components);
  vec_free(mix->logits);
  vec_free(mix->weights);
  free(mix);
}

/* Initialize all other component means from one component plus Gaussian jitter. */
void mixmvn_init_jitter_from_component(mixture_MVN *mix, int component,
                                  double jitter_sd) {
  int i, k;
  Vector *mu;

  /* Initialize the mean vector with the mean of the component to be copied. */
  mu = vec_new(mix->components[component]->n * mix->components[component]->d);
  mmvn_save_mu(mix->components[component], mu);

  for (k = 0; k < mix->ncomponents; k++) {
    /* Skip the specified component. */
    if (k == component)
      continue;

    /* Add Gaussian jitter element-wise to the mean vector, store the jittered
      vector, and restore the original. */
    for (i = 0; i < mu->size; i++)
      vec_set(mu, i, vec_get(mu, i) + norm_draw(0, jitter_sd));

    mmvn_set_mu(mix->components[k], mu);
    mmvn_save_mu(mix->components[component], mu);
  }

  vec_free(mu);
}

/* Update each component with the shared covariance parameters. */
void mixmvn_update_covariance(mixture_MVN *mix, CovarData *data) {
  int k;

  for (k = 0; k < mix->ncomponents; k++)
    update_covariance(mix->components[k], data, k);
}

/* Normalize unconstrained component logits into mixture probabilities. */
void mixmvn_update_weights(mixture_MVN *mix) {
  int k;
  double max_logit = -INFINITY, sum_exp = 0.0;

  /* Find the maximum logit. */
  for (k = 0; k < mix->ncomponents; k++) {
      if (vec_get(mix->logits, k) > max_logit)
        max_logit = vec_get(mix->logits, k);
  }

  /* Compute the sum of exponentials of the centered logits. */
  for (k = 0; k < mix->ncomponents; k++)
    sum_exp += exp(vec_get(mix->logits, k) - max_logit);

  /* Normalize the logits and compute the mixture weights. */
  for (k = 0; k < mix->ncomponents; k++)
    vec_set(mix->weights, k, exp(vec_get(mix->logits, k) - max_logit) / sum_exp);
}

/* Sample a component from the mixture by drawing uniformly in [0, 1) and
    then finding the point in the cumulative distribution where the draw falls. */
int mixmvn_sample_component(mixture_MVN *mix) {
  int component;
  double draw = unif_rand(), cdf = 0.0;

  for (component = 0; component < mix->ncomponents; component++) {
    cdf += vec_get(mix->weights, component);
    if (draw <= cdf)
      return component;
  }

  return mix->ncomponents - 1;
}

/* Compute the log density of the mixture via log-sum-exp. */
double mixmvn_log_dens(mixture_MVN *mix, Vector *x) {
  int k;
  double max_ldens = -INFINITY, sum_exp = 0;
  double *ldens;

  ldens = smalloc(mix->ncomponents * sizeof(double));

  /* Compute the log density of each component and find the maximum. */
  for (k = 0; k < mix->ncomponents; k++) {
    ldens[k] = log(vec_get(mix->weights, k)) +
      mmvn_log_dens(mix->components[k], x);
    if (ldens[k] > max_ldens)
      max_ldens = ldens[k];
  }

  /* Compute the sum of exponentials of the centered log densities. */
  for (k = 0; k < mix->ncomponents; k++)
    sum_exp += exp(ldens[k] - max_ldens);

  free(ldens);

  /* log-sum-exp */
  return max_ldens + log(sum_exp);
}

void mixmvn_print_embedding(mixture_MVN *mix, char **names, int n, int d,
                            FILE *F) {
  int k;
  char prefix[32];

  for (k = 0; k < mix->ncomponents; k++) {
    if (mix->ncomponents > 1) {
      snprintf(prefix, sizeof(prefix), "%d", k);
      mmvn_print_embedding(mix->components[k], names, n, d, prefix, F);
    }
    else
      mmvn_print_embedding(mix->components[k], names, n, d, NULL, F);
  }
}
