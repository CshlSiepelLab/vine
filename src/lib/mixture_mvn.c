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
#include <phast/misc.h>
#include <covariance.h>
#include <mixture_mvn.h>


/* Allocate memory and initialize each component mvn. */
mixture_MVN *mixmvn_new(int ncomponents, int n, int d, enum mvn_type type) {
  int k;
  mixture_MVN *mix;

  if (ncomponents <= 0)
    die("ERROR in mixmvn_new: number of mixture components must be positive.\n");

  mix = smalloc(sizeof(mixture_MVN));
  mix->ncomponents = ncomponents;
  mix->components = smalloc(ncomponents * sizeof(multi_MVN*));

  for (k = 0; k < ncomponents; k++)
    mix->components[k] = mmvn_new(n, d, type);

  return mix;
}

/* Wrapper to free each component mvn. */
void mixmvn_free(mixture_MVN *mix) {
  int k;

  for (k = 0; k < mix->ncomponents; k++)
    if (mix->components[k] != NULL)
      mmvn_free(mix->components[k]);

  free(mix->components);
  free(mix);
}

/* Return a pointer to the specified component. */
multi_MVN *mixmvn_get_component(mixture_MVN *mix, int component) {
  return mix->components[component];
}

/* Initialize all other component means from one component plus Gaussian jitter. */
void mixmvn_init_jitter_from_component(mixture_MVN *mix, int component,
                                  double jitter_sd) {
  int i, k;
  Vector *mu;

  assert(component >= 0 && component < mix->ncomponents);
  assert(jitter_sd >= 0);

  /* Initialize the mean vector with the mean of the specified component. */
  mu = vec_new(mix->components[component]->n * mix->components[component]->d);
  mmvn_save_mu(mix->components[component], mu);

  for (k = 0; k < mix->ncomponents; k++) {
    /* Skip the specified component. */
    if (k == component)
      continue;

    /* Add Gaussian jitter to each element of the mean vector. */
    for (i = 0; i < mu->size; i++)
      vec_set(mu, i, vec_get(mu, i) + norm_draw(0, jitter_sd));

    /* Store the jittered mean vector. */
    mmvn_set_mu(mix->components[k], mu);

    /* Restore the mean vector of the specified component for the next component. */
    mmvn_save_mu(mix->components[component], mu);
  }

  vec_free(mu);
}

/* Update each component with the shared covariance parameters. */
void mixmvn_update_covariance(mixture_MVN *mix, CovarData *data) {
  int k;

  for (k = 0; k < mix->ncomponents; k++)
    nj_update_covariance(mixmvn_get_component(mix, k), data, k);
}

/* Sample a component from the mixture. */
int mixmvn_sample_component(mixture_MVN *mix) {
  int component;

  /* Sample a component from the mixture, truncating down the random draw to nearest int. */
  component = (int)(unif_rand() * mix->ncomponents);

  /* Handle the edge case where the random draw is 1.0 component equals the number of components 
      which is out of bounds in our 0-based indexing */
  if (component >= mix->ncomponents)
    component = mix->ncomponents - 1;

  return component;
}

/* Wrapper to compute the log density of the mixture via log-sum-exp,
assuming equal mixture weights here. */
double mixmvn_log_dens(mixture_MVN *mix, Vector *x) {
  int k;
  double max_ldens = -INFINITY, sum_exp = 0;
  double *ldens;

  ldens = smalloc(mix->ncomponents * sizeof(double));

  /* Compute the log density of each component and find the maximum. */
  for (k = 0; k < mix->ncomponents; k++) {
    ldens[k] = mmvn_log_dens(mix->components[k], x);
    if (ldens[k] > max_ldens)
      max_ldens = ldens[k];
  }

  /* Compute the sum of exponentials of the centered log densities. */
  for (k = 0; k < mix->ncomponents; k++)
    sum_exp += exp(ldens[k] - max_ldens);

  free(ldens);

  /* log-sum-exp */
  return max_ldens + log(sum_exp) - log((double)mix->ncomponents);
}
