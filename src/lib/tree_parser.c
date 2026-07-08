/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* shared utilities for reading collections of Newick trees from file */

#include <phast/misc.h>
#include <phast/trees.h>
#include <phast/stringsplus.h>
#include "tree_parser.h"

List *tr_read_trees_from_file(const char *fname) {
  FILE *f = phast_fopen(fname, "r");
  String *line = str_new(STR_VERY_LONG_LEN);
  List *trees = lst_new_ptr(1000);
  int lineno = 0;
  while (str_readline(line, f) != EOF) {
    str_double_trim(line);
    if (line->length == 0) continue;
    lineno++;
    if (line->chars[0] != '(')
      die("ERROR in line %d of %s: Input does not look like a Newick-formatted tree.\n",
          lineno, fname);
    if (line->chars[line->length-1] == ';')
      line->chars[--line->length] = '\0';
    lst_push_ptr(trees, tr_new_from_string(line->chars));
  }
  str_free(line);
  fclose(f);
  return trees;
}
