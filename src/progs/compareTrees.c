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
#include <phast/lists.h>
#include <openblas/cblas.h>
#include <kl.h>
#include <tree_parser.h>
#include "compareTrees.help"

int main(int argc, char *argv[]) {
  int opt_idx, c;
  char *est_fname, *ref_fname;
  List *trees_est, *trees_ref;
  double split_kl, embed_kl;
  int dim = -1;   /* -1 means "use vine's default formula" */
  int nthreads = 1;

  struct option long_opts[] = {
    {"dim", 1, 0, 'd'},
    {"nthreads", 1, 0, 'n'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  while ((c = getopt_long(argc, argv, "d:n:h", long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'd':
      dim = atoi(optarg);
      if (dim < 1)
        die("ERROR: --dim must be a positive integer.\n");
      break;
    case 'n':
      nthreads = atoi(optarg);
      if (nthreads < 1)
        die("ERROR: --nthreads must be a positive integer.\n");
      break;
    case 'h':
      printf("%s", HELP);
      exit(0);
    case '?':
      die("Bad argument.  Try 'compareTrees -h'.\n");
    }
  }

  if (optind != argc - 2)
    die("Missing required arguments.  Try '%s -h'.\n", argv[0]);

  openblas_set_num_threads(nthreads);

  est_fname = argv[optind];
  ref_fname = argv[optind + 1];

  fprintf(stderr, "Reading trees from %s...\n", est_fname);
  trees_est = tr_read_trees_from_file(est_fname);

  fprintf(stderr, "Reading trees from %s...\n", ref_fname);
  trees_ref = tr_read_trees_from_file(ref_fname);

  fprintf(stderr, "Computing split KL divergence (%d vs %d trees)...\n",
          lst_size(trees_est), lst_size(trees_ref));
  tr_split_kl(trees_est, trees_ref, &split_kl);

  fprintf(stderr, "Computing embedding KL divergence...\n");
  tr_embed_kl(trees_est, trees_ref, dim, &embed_kl);

  printf("Successfully processed %d trees from %s and %d trees from %s.\n",
         lst_size(trees_est), est_fname, lst_size(trees_ref), ref_fname);
  printf("Mean KL divergences of %s (estimate) from %s (reference):\n",
         est_fname, ref_fname);
  printf("Mean_split_KL: %f\n", split_kl);
  printf("Mean_embed_KL: %f\n", embed_kl);

  return 0;
}
