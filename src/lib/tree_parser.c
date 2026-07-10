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

TreeNode *tr_parse_newick_line(String *line, const char *fname, int lineno) {
  str_double_trim(line);
  if (line->length == 0)
    return NULL;

  if (line->chars[0] != '(')
    die("ERROR in line %d of %s: Input does not look like a Newick-formatted tree.\n",
        lineno, fname);

  /* tr_new_from_string expects a string without the trailing semicolon */
  if (line->chars[line->length - 1] == ';')
    line->chars[--line->length] = '\0';

  return tr_new_from_string(line->chars);
}

/* input file should have one newick tree 
    per line */
List *tr_read_trees_from_file(const char *fname) {
  FILE *f = phast_fopen(fname, "r");
  String *line = str_new(STR_VERY_LONG_LEN);
  /* set initial capacity that is auto allocated larger as needed */
  List *trees = lst_new_ptr(1000);
  int lineno = 0;
  while (str_readline(line, f) != EOF) {
    lineno++;
    TreeNode *tree = tr_parse_newick_line(line, fname, lineno);
    if (tree != NULL)
      lst_push_ptr(trees, tree);
  }
  str_free(line);
  fclose(f);
  return trees;
}
