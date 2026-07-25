/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <phast/misc.h>
#include <nj.h>
#include <likelihoods.h>
#include <backprop.h>
#include <phast/sufficient_stats.h>
#include <sparse_matrix.h>
#include <phast/lists.h>
#include <phast/msa.h>
#include <phast/sufficient_stats.h>
#include <phast/subst_mods.h>
#include <phast/tree_model.h>
#include <rf.h>
#include <branch_score.h>
#include <geometry.h>
#include <tree_parser.h>
#include "evalTrees.help"

/* OpenBLAS thread control.  Declared here (rather than pulling in
 * <openblas/cblas.h>, whose location is not portable) and compiled only
 * when the linked BLAS actually provides the symbol.  VINE_HAVE_OPENBLAS
 * is defined by CMake; BLAS libraries that lack it (e.g. Apple Accelerate)
 * simply skip the thread-limiting call below. */
#ifdef VINE_HAVE_OPENBLAS
extern void openblas_set_num_threads(int);
#endif

static inline
void print_stats(FILE *F, double mean, double stdev, double median,
                 double min, double max, double min_95CI, double max_95CI,
                 double q25, double q75) {
  fprintf(F, "Mean: %f\n", mean);
  fprintf(F, "Std: %f\n", stdev);
  fprintf(F, "Median: %f\n", median);
  fprintf(F, "Range: %f - %f\n", min, max);
  fprintf(F, "95%%_CI: %f - %f\n", min_95CI, max_95CI);
  fprintf(F, "50%%_CI: %f - %f\n", q25, q75);
}

int main(int argc, char *argv[]) {
#ifdef VINE_HAVE_OPENBLAS
  openblas_set_num_threads(1);
#endif

  TreeNode *tree;
  TreeModel *mod = NULL;
  double kappa = -1, ll;
  String *line = str_new(STR_VERY_LONG_LEN);
  int opt_idx, lineno = 0, input_lineno = 0, i, j, nleaves = 0, npairs = 0;
  CovarData *data;
  int c;  /* getopt_long returns int; storing in char makes EOF (-1)
             alias to 255 on platforms where char is unsigned, so the
             option loop never terminates. */
  FILE *treefile, *msafile = NULL, *rf_matrix_outfile = NULL,
    *rf_mds_outfile = NULL;
  MarkovMatrix *rmat;
  msa_format_type format;
  TreeNode *topol_ref = NULL;
  TreeNode *bsd_ref = NULL;
  MSA *evalaln = NULL;
  double mean, stdev, median, min, max, min_95CI, max_95CI, q25, q75;
  char *topolfname = NULL, *msafname = NULL, *treefname = NULL, *bsdref_fname = NULL;
  char *rf_matrix_outfname = NULL, *rf_mds_outfname = NULL;
  List *rfdists = NULL, *lldists, *bsddists = NULL;
  char **names = NULL;
  Matrix *D = NULL;
  List **Dij_list = NULL;
  int is_crispr = FALSE;
  int do_entropy = FALSE;
  int do_rf_matrix = FALSE;
  int do_rf_mds = FALSE;
  List *trees_all = NULL;
  CrisprMutTable *crispr_muts = NULL;
  CrisprMutModel *crispr_mod = NULL;
  enum crispr_model_type crispr_modtype = SITEWISE;
  enum crispr_mutrates_type crispr_muttype = UNIF;
  
  struct option long_opts[] = {
    {"hky-kappa", 1, 0, 'k'},
    {"crispr", 0, 0, 'c'},
    {"entropy", 0, 0, 'e'},
    {"rf-matrix", 1, 0, 'r'},
    {"rf-mds", 1, 0, 'M'},
    {"tree-model", 1, 0, 'm'},
    {"model-fit", 1, 0, 'f'},
    {"topology", 1, 0, 't'},
    {"branch-score", 1, 0, 'b'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  while ((c = getopt_long(argc, argv, "b:ef:k:m:t:chr:M:",
                          long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'k':
      kappa = atof(optarg);
      if (kappa < 0)
        die("ERROR: --hky-kappa must be > 0.\n");
      break;
    case 'm':
      mod = tm_new_from_file(phast_fopen(optarg, "r"), 1);
      break;
    case 'f':
      msafname = optarg;
      msafile = phast_fopen(optarg, "r");
      break;
    case 't':
      topolfname = optarg;
      topol_ref = tr_new_from_file(phast_fopen(topolfname, "r"));
      break;
    case 'b':
      bsdref_fname = optarg;
      bsd_ref = tr_new_from_file(phast_fopen(bsdref_fname, "r"));
      break;
    case 'c':
      is_crispr = TRUE;
      break;
    case 'e':
      do_entropy = TRUE;
      break;
    case 'r':
      do_rf_matrix = TRUE;
      rf_matrix_outfname = optarg;
      break;
    case 'M':
      do_rf_mds = TRUE;
      rf_mds_outfname = optarg;
      break;
    case 'h':
      printf("%s", HELP); 
      exit(0);
    case '?':
      die("Bad argument.  Try 'evalTrees -h'.\n");
    }
  }

  if (optind != argc - 1)
    die("Missing required argument.  Try '%s -h'.\n", argv[0]);

    /* open dna or crispr file */
  if (is_crispr == TRUE) {
    if (msafile == NULL)
      die("Option --crispr requires --model-fit.\n");
    crispr_muts = cpr_read_table(msafile);
  }
  else if (msafile != NULL) {
    format = msa_format_for_content(msafile, 1);
    evalaln = msa_new_from_file_define_format(msafile, format, DEFAULT_ALPHABET);
  }
  
  if (do_entropy && (topol_ref != NULL || msafname != NULL || is_crispr))
    die("Option --entropy cannot be combined with --topology, --model-fit, or --crispr.\n");

  if ((do_rf_matrix || do_rf_mds) &&
      (topol_ref != NULL || msafname != NULL || is_crispr ||
       bsd_ref != NULL || do_entropy))
    die("Options --rf-matrix and --rf-mds cannot be combined with other evaluation modes.\n");

  if (bsd_ref != NULL && (topol_ref != NULL || msafname != NULL || is_crispr || do_entropy))
    die("Option --branch-score cannot be combined with --topology, --model-fit, --crispr, or --entropy.\n");

  if (evalaln == NULL && (mod != NULL || kappa > 0))
    die("Options --tree-model and --hky-kappa require --model-fit.\n");
  
  if (evalaln != NULL && mod != NULL && kappa > 0)
    die("Options --tree-model and --hky-kappa are mutually exclusive.\n");

  if (evalaln != NULL && topol_ref != NULL)
    die("Options --model-fit and --topology are mutually exclusive.\n");

  if (is_crispr && (mod != NULL || kappa > 0))
    die("Options --tree-model and --hky-kappa are incompatible with --crispr.\n");
  
  /* open tree file */
  treefname = argv[optind];
  fprintf(stderr, "Reading trees from %s...\n", treefname);
  treefile = phast_fopen(treefname, "r");  
  
  /* set up for --model-fit */
  if (evalaln != NULL || is_crispr == TRUE) {
    fprintf(stderr, "Evaluating model fit on %s...\n", msafname);

    if (evalaln != NULL && evalaln->ss == NULL)
      ss_from_msas(evalaln, 1, TRUE, NULL, NULL, NULL, -1, 0);
    else if (is_crispr)
      crispr_mod = cpr_new_model(crispr_muts, NULL, crispr_modtype, crispr_muttype);
      
    /* this is mostly a dummy; only the msa or crispr mod is used */
    D = mat_new(5, 5);
    data = new_covar_data(CONST, D, 1, evalaln, crispr_mod, NULL, FALSE,
                             1.0, 3, 1.0, FALSE, -1, FALSE, FALSE, FALSE,
                             NULL, NULL, FALSE);
    lldists = lst_new_dbl(1000);
  }
  else if (topol_ref != NULL) {
    rfdists = lst_new_dbl(1000);
    fprintf(stderr, "Evaluating RF distance to %s...\n", topolfname);
  }
  else if (bsd_ref != NULL) {
    bsddists = lst_new_dbl(1000);
    trees_all = lst_new_ptr(1000);   /* retained for the point (mean-tree) BSD */
    fprintf(stderr, "Evaluating branch-score distance to %s...\n", bsdref_fname);
  }
  else if (do_entropy) {
    trees_all = lst_new_ptr(1000);
    fprintf(stderr, "Computing split entropy...\n");
  }
  else if (do_rf_matrix || do_rf_mds) {
    trees_all = lst_new_ptr(1000);
    if (do_rf_matrix)
      rf_matrix_outfile = phast_fopen(rf_matrix_outfname, "w");
    if (do_rf_mds)
      rf_mds_outfile = phast_fopen(rf_mds_outfname, "w");
    fprintf(stderr, "Computing pairwise RF distances...\n");
  }
  else
    fprintf(stderr, "Computing pairwise-distance stats...\n");
  
  while (str_readline(line, treefile) != EOF) {
    input_lineno++;
    tree = tr_parse_newick_line(line, treefname, input_lineno);
    if (tree == NULL)
      continue;

    lineno++;

    if (evalaln != NULL || is_crispr) {
      if (mod == NULL) { /* do this the first time through; need a tree to initialize */        
        rmat = mm_new(strlen(DEFAULT_ALPHABET), DEFAULT_ALPHABET, CONTINUOUS);
        mod = tm_new(tree, rmat, NULL, kappa > 0 ? HKY85 : JC69, DEFAULT_ALPHABET,
                     1, 1, NULL, -1);
        if (is_crispr) {  /* tree model just a dummy in this case */
          crispr_mod->mod = mod;
          cpr_prep_model(crispr_mod);
        }
        else {
          tm_init_backgd(mod, evalaln, -1);
          if (kappa > 0) /* create HKY model */ {
            fprintf(stderr, "Using HKY85 with kappa = %f...\n", kappa);
            tm_set_HKY_matrix(mod, kappa, -1);
          }
          else {           /* create JC model */
            fprintf(stderr, "Using JC69...\n");
            tm_set_JC69_matrix(mod);
          }
        }
      }
      else
        reset_tree_model(mod, tree);

      if (evalaln != NULL) {
        /* have to force index rebuild because node ids can change */
        sfree(mod->msa_seq_idx);
        tm_build_seq_idx(mod, evalaln);
        ll = compute_log_likelihood(mod, data, NULL);
      }
      else { /* crispr case */
        sfree(crispr_mod->mod->msa_seq_idx);
        cpr_build_seq_idx(crispr_mod->mod, crispr_mod->mut);
        ll = cpr_compute_log_likelihood(data->crispr_mod, NULL);
      }

      /* occasionally get -inf from 0-length branches; let's just
         ignore those for now */
      if (isfinite(ll))  
        lst_push_dbl(lldists, ll);
    }

    else if (topol_ref != NULL) {
      double d = tr_robinson_foulds(tree, topol_ref);
      lst_push_dbl(rfdists, d);
      tr_free(tree);
    }

    else if (bsd_ref != NULL) {
      double d = tr_branch_score(tree, bsd_ref);
      lst_push_dbl(bsddists, d);
      lst_push_ptr(trees_all, tree);   /* retain for point (mean-tree) BSD */
    }

    else if (do_entropy || do_rf_matrix || do_rf_mds) {
      lst_push_ptr(trees_all, tree);
    }

    else {  /* collect distance statistics */
      if (names == NULL) {  /* first time through get canonical list
                               of leaf names */
        List *l = tr_leaf_names(tree);
        lst_qsort_str(l, ASCENDING);
        nleaves = lst_size(l);
        npairs = nleaves * (nleaves-1) / 2;
        names = smalloc(nleaves * sizeof(char*)); 
        for (i = 0; i < lst_size(l); i++) {
          String *s = lst_get_ptr(l, i);
          names[i] = s->chars;
        }
        /* also set up lists of pairwise distances across trees */
        Dij_list = smalloc(npairs * sizeof(List*));
        for (i = 0; i < nleaves; i++) 
          for (j = i+1; j < nleaves; j++)
            Dij_list[nj_i_j_to_dist(i, j, nleaves)] = lst_new_dbl(1000);
        fprintf(stderr, "Extracting pairwise distances for all trees...\n");
      }
      
      /* get distance matrix for all pairs of leaves for this tree */
      D = tree_to_distances(tree, names, nleaves);
      /* add distances to corresponding lists */
      for (i = 0; i < nleaves; i++) {
        for (j = i+1; j < nleaves; j++) {
          double d = mat_get(D, i, j);
          lst_push_dbl(Dij_list[nj_i_j_to_dist(i, j, nleaves)], d);
        }
      }
      
      mat_free(D);
      tr_free(tree);
    }
  }

  /* output results */
  // fprintf(stderr, "Done processing %d trees.\n", lineno);
  if (lineno == 0)
    die("ERROR: no trees found in %s.\n", treefname);
   
  if (evalaln != NULL || is_crispr) {
    printf("Successfully processed %d trees from %s.\n", lineno, treefname);
    printf("Log likelihood evaluated on %s:\n", msafname);
    lst_dbl_stats(lldists, &mean, &stdev, &median, &min, &max,
                  &min_95CI, &max_95CI, &q25, &q75);
    print_stats(stdout, mean, stdev, median, min, max, min_95CI,
                max_95CI, q25, q75);
    if (evalaln != NULL)
      printf("Mean per site: %f\n", mean/evalaln->length);
  }
  else if (topol_ref != NULL) {
    printf("Successfully processed %d trees from %s.\n", lineno, treefname);
    lst_dbl_stats(rfdists, &mean, &stdev, &median, &min, &max,
                  &min_95CI, &max_95CI, &q25, &q75);
    printf("Robinson Foulds distances against %s:\n", topolfname);
    print_stats(stdout, mean, stdev, median, min, max, min_95CI,
                max_95CI, q25, q75);
  }
  else if (bsd_ref != NULL) {
    printf("Successfully processed %d trees from %s.\n", lineno, treefname);
    lst_dbl_stats(bsddists, &mean, &stdev, &median, &min, &max,
                  &min_95CI, &max_95CI, &q25, &q75);
    printf("Branch-score distances against %s:\n", bsdref_fname);
    print_stats(stdout, mean, stdev, median, min, max, min_95CI,
                max_95CI, q25, q75);
    printf("Point (posterior-mean-tree) BSD: %f\n",
           tr_branch_score_pointest(trees_all, bsd_ref));
    printf("Reference tree length: %f\n", tr_tree_length(bsd_ref));
  }
  else if (do_entropy) {
    int n_topologies;
    double H_split, H_top, mean_var, mean_var_per_branch;
    tr_tree_entropy(trees_all, &n_topologies, &H_split, &H_top,
                    &mean_var, &mean_var_per_branch);
    printf("Unique topologies: %d\n", n_topologies);
    printf("Split entropy: %f\n", H_split);
    printf("Topology entropy: %f\n", H_top);
    printf("Mean branch-length variance: %f\n", mean_var_per_branch);
  }
  else if (do_rf_matrix || do_rf_mds) {
    D = tr_robinson_foulds_matrix(trees_all, TRUE);

    if (do_rf_matrix)
      tr_write_robinson_foulds_matrix(D, rf_matrix_outfile);

    if (do_rf_mds) {
      Matrix *points;
      if (lineno < 3)
        die("ERROR: --rf-mds requires at least three trees.\n");
      points = classical_mds(D, 2);
      fprintf(rf_mds_outfile, "tree\tx\ty\n");
      for (i = 0; i < lineno; i++)
        fprintf(rf_mds_outfile, "tree%d\t%.6f\t%.6f\n", i + 1,
                mat_get(points, i, 0), mat_get(points, i, 1));
      mat_free(points);
    }

    mat_free(D);
    if (rf_matrix_outfile != NULL)
      fclose(rf_matrix_outfile);
    if (rf_mds_outfile != NULL)
      fclose(rf_mds_outfile);
  }
  else {
    printf("#leaf1\tleaf2\tmean\tstd\tmed\tmin\tmax\tlow95CI\thigh95CI\tlow50CI\thigh50CI\n");
    for (i = 0; i < nleaves; i++) {
      for (j = i+1; j < nleaves; j++) {
        lst_dbl_stats(Dij_list[nj_i_j_to_dist(i, j, nleaves)], &mean, &stdev,
                      &median, &min, &max, &min_95CI, &max_95CI, &q25, &q75);
        /* in this case, print a table */
        printf("%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
               names[i], names[j], mean, stdev, median, min, max, min_95CI,
               max_95CI, q25, q75);
      }
    }
  }
}


  
