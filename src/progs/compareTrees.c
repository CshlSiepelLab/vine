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
#include <rf.h>
#include <tree_parser.h>
#include "compareTrees.help"

int main(int argc, char *argv[]) {
  int opt_idx, c;
  char *est_fname, *ref_fname;
  List *trees_est, *trees_ref;
  double mean_kl;

  struct option long_opts[] = {
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  while ((c = getopt_long(argc, argv, "h", long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'h':
      printf("%s", HELP);
      exit(0);
    case '?':
      die("Bad argument.  Try 'compareTrees -h'.\n");
    }
  }

  if (optind != argc - 2)
    die("Missing required arguments.  Try '%s -h'.\n", argv[0]);

  est_fname = argv[optind];
  ref_fname = argv[optind + 1];

  fprintf(stderr, "Reading trees from %s...\n", est_fname);
  trees_est = tr_read_trees_from_file(est_fname);

  fprintf(stderr, "Reading trees from %s...\n", ref_fname);
  trees_ref = tr_read_trees_from_file(ref_fname);

  fprintf(stderr, "Computing split KL divergence (%d vs %d trees)...\n",
          lst_size(trees_est), lst_size(trees_ref));

  tr_split_kl(trees_est, trees_ref, &mean_kl);

  printf("Successfully processed %d trees from %s and %d trees from %s.\n",
         lst_size(trees_est), est_fname, lst_size(trees_ref), ref_fname);
  printf("Mean split KL divergence of %s (estimate) from %s (reference):\n",
         est_fname, ref_fname);
  printf("Mean_split_KL: %f\n", mean_kl);

  return 0;
}
