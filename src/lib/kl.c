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

#include <math.h>
#include <phast/lists.h>
#include <phast/trees.h>
#include <phast/misc.h>
#include <phast/stringsplus.h>
#include <nj.h>
#include <geometry.h>
#include <multi_mvn.h>
#include <rf.h>
#include <vine.h>
#include "kl.h"


/* Compute the mean KL divergence between the split-frequency distributions of
   two tree samples. Split inclusion probabilities are estimated from observed
   frequencies with Laplace smoothing, the Bernoulli KL divergence is computed
   for each split in the union of the samples, and the result is averaged
   across splits. */
void tr_split_kl(List *trees_est, List *trees_ref, double *mean_kl) {
  *mean_kl = 0.0;

  int S_est = lst_size(trees_est), S_ref = lst_size(trees_ref);
  if (S_est == 0 || S_ref == 0)
    die("ERROR in tr_split_kl: at least one input tree list is empty.\n");

  /* get the two sorted lists of splits and their counts */
  int nsc_e;  /* number of split counts in estimate */
  int nsc_r;  /* number of split counts in reference */
  int n_e;    /* number of leaves in estimate */
  int n_r;    /* number of leaves in reference */
  SplitCount *sce = tr_collect_split_counts(trees_est, &nsc_e, &n_e);
  SplitCount *scr = tr_collect_split_counts(trees_ref, &nsc_r, &n_r);

  if (n_e != n_r)
    die("ERROR in tr_split_kl: estimate and reference trees have different numbers of leaves.\n");

  /* walk through the two sorted lists of splits to compare them */
  int i = 0, j = 0, nsplits = 0;
  double sum_kl = 0.0;
  while (i < nsc_e || j < nsc_r) {
    /* handle mismatched split lists to prevent out of bounds errors */
    int cmp;
    if (i >= nsc_e) cmp = 1;
    else if (j >= nsc_r) cmp = -1;
    else cmp = tr_bitmask_cmp(sce[i].mask, scr[j].mask);

    /* if the split exists in only one sample then the other becomes zero*/
    int c_e = (cmp <= 0) ? sce[i].count : 0;
    int c_r = (cmp >= 0) ? scr[j].count : 0;

    /* Convert counts to Laplace-smoothed inclusion probabilities so no 
        split has p or q exactly 0 or 1 (which would blow up the KL divergence below). */
    double p = (c_e + 0.5) / (S_est + 1.0);
    double q = (c_r + 0.5) / (S_ref + 1.0);
    double kl = p * log(p / q) + (1.0 - p) * log((1.0 - p) / (1.0 - q));  /* Bernoulli KL divergence */
    sum_kl += kl; /* accumulate the marginal split-frequency KL divergence */
    nsplits++;

    /* increment while respecting the mismatched split lists 
        to prevent out of bounds errors */
    if (cmp <= 0) i++;
    if (cmp >= 0) j++;
  }

  /* compute the mean KL divergence per split */
  *mean_kl = (nsplits > 0) ? sum_kl / nsplits : 0.0;

  for (int k = 0; k < nsc_e; k++) tr_bitmask_free(sce[k].mask);
  for (int k = 0; k < nsc_r; k++) tr_bitmask_free(scr[k].mask);
  sfree(sce); sfree(scr);
}

/* Turn one tree into a fixed-length vector of pairwise distances after
    Euclidean embedding. */
static Vector *embed_tree_pairwise_dists(TreeNode *tree, char **names, int n, int dim) {
  Matrix *D = tree_to_distances(tree, names, n);  /* cophenetic distance matrix */

  /* MDS to get the Eublidean embedding from the pairwise distances */
  CovarData *data = new_covar_data(CONST, D, dim, NULL, NULL, names,
                                    FALSE, 1.0, 3, 1.0, FALSE, 1.0, FALSE,
                                    FALSE, FALSE, NULL, NULL, FALSE);
  multi_MVN *mmvn = mmvn_new(n, dim, MVN_DIAG);
  estimate_mmvn_from_distances(data, mmvn);

  /* Get pairwise distances from the embedding coordinates */
  Vector *mu = mmvn->mvn->mu;
  Vector *dists = vec_new(n * (n - 1) / 2); /* one dist per unique leaf pair */
  int idx = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {

      /* Euclidean distance */
      double ssq = 0.0;
      for (int k = 0; k < dim; k++) {
        double diff = vec_get(mu, i*dim + k) - vec_get(mu, j*dim + k);
        ssq += diff * diff;
      }
      vec_set(dists, idx++, sqrt(ssq));
    }
  }

  mmvn_free(mmvn);
  free_covar_data(data);   /* also frees D */
  return dists;
}

/* Accumulate per-pair mean and standard deviation of embedded 
    pairwise distances across a set of trees */
static void embed_pair_stats(List *trees, char **names, int n, int dim,
                              double **mean_out, double **sd_out) {
  int S = lst_size(trees);
  int npairs = n * (n - 1) / 2;
  double *sum = calloc(npairs, sizeof(double)); /* first moment */
  double *sumsq = calloc(npairs, sizeof(double)); /* second moment */

  for (int s = 0; s < S; s++) {
    TreeNode *t = lst_get_ptr(trees, s);

    /* embed the tree into a Euclidean space and get pairwise distances */
    Vector *d = embed_tree_pairwise_dists(t, names, n, dim);

    /* accumulate moments */
    for (int k = 0; k < npairs; k++) {
      double v = vec_get(d, k);
      sum[k] += v;
      sumsq[k] += v * v;
    }
    vec_free(d);
  }

  /* compute mean and standard deviation */
  double *mean = smalloc(npairs * sizeof(double));
  double *sd = smalloc(npairs * sizeof(double));
  for (int k = 0; k < npairs; k++) {
    mean[k] = sum[k] / S;
    double var = sumsq[k] / S - mean[k] * mean[k];
    if (var < 1e-12) var = 1e-12;   /* floor to avoid degenerate KL below */
    sd[k] = sqrt(var);
  }
  free(sum); free(sumsq);

  *mean_out = mean;
  *sd_out = sd;
}

/* Compare two tree samples by embedding each tree into Euclidean space,
    modeling each embedded leaf-pair distance as a univariate Gaussian,
    and averaging the resulting per-pair Gaussian KL divergences. */
void tr_embed_kl(List *trees_est, List *trees_ref, int dim, double *mean_kl) {
  *mean_kl = 0.0;
  int S_est = lst_size(trees_est), S_ref = lst_size(trees_ref);
  if (S_est == 0 || S_ref == 0)
    die("ERROR in tr_split_kl: at least one input tree list is empty.\n");

  /* Get the sorted list of leaf names from the first tree */
  TreeNode *t0 = lst_get_ptr(trees_est, 0);
  List *namelist = tr_leaf_names(t0);
  lst_qsort_str(namelist, ASCENDING);
  int n = lst_size(namelist);
  if (n < 3) { lst_free_strings(namelist); lst_free(namelist); return; }

  char **names = smalloc(n * sizeof(char*));
  for (int i = 0; i < n; i++)
    names[i] = ((String*)lst_get_ptr(namelist, i))->chars;

  /* Use vine's default dimensionality if not specified */
  if (dim <= 0) dim = vine_default_dim(n);

  /* Get the mean and sd for each pair in each tree list */
  double *mean_e, *sd_e, *mean_r, *sd_r;
  embed_pair_stats(trees_est, names, n, dim, &mean_e, &sd_e);
  embed_pair_stats(trees_ref, names, n, dim, &mean_r, &sd_r);

  int npairs = n * (n - 1) / 2;
  double sum_kl = 0.0;  
  for (int k = 0; k < npairs; k++) {
    /* closed form Gaussian KL divergence */
    double kl = log(sd_r[k] / sd_e[k])
              + (sd_e[k]*sd_e[k] + (mean_e[k]-mean_r[k])*(mean_e[k]-mean_r[k])) / (2.0*sd_r[k]*sd_r[k])
              - 0.5;
    sum_kl += kl;
  }

  /* compute mean KL divergence per pair */
  *mean_kl = (npairs > 0) ? sum_kl / npairs : 0.0;

  sfree(names);
  lst_free_strings(namelist);
  lst_free(namelist);
  free(mean_e); free(sd_e); free(mean_r); free(sd_r);
}
