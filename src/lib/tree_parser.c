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

/* write TAXA block listing the leaf names of tree */
static void print_nexus_taxa_block(TreeNode *tree, FILE *outf) {
  List *trav = tree->nodes;
  int ntaxa = 0;
  for (int i = 0; i < tree->nnodes; i++) {
    TreeNode *n = lst_get_ptr(trav, i);
    if (n->lchild == NULL && n->rchild == NULL)
      ntaxa++;
  }
  fprintf(outf, "BEGIN TAXA;\n");
  fprintf(outf, "  DIMENSIONS NTAX=%d;\n", ntaxa);
  fprintf(outf, "  TAXLABELS\n");
  for (int i = 0; i < tree->nnodes; i++) {
    TreeNode *n = lst_get_ptr(trav, i);
    if (n->lchild == NULL && n->rchild == NULL)
      fprintf(outf, "    %s\n", n->name);
  }
  fprintf(outf, "  ;\nEND;\n\n");
}

void tr_print_nexus(TreeNode *tree, FILE *outf) {
  fprintf(outf, "#NEXUS\n\n");
  print_nexus_taxa_block(tree, outf);
  fprintf(outf, "BEGIN TREES;\n");
  fprintf(outf, "  TREE TREE1 = [&R] ");
  tr_print(outf, tree, /*show_branch_lengths=*/1);
  fprintf(outf, "END;\n\n");
}

void tr_print_set_nexus(List *tree_lst, FILE *outf) {
  TreeNode *tree0 = (TreeNode*)lst_get_ptr(tree_lst, 0);

  fprintf(outf, "#NEXUS\n\n");
  print_nexus_taxa_block(tree0, outf);

  fprintf(outf, "BEGIN TREES;\n");
  for (int s = 0; s < lst_size(tree_lst); s++) {
    TreeNode *tree = (TreeNode*)lst_get_ptr(tree_lst, s);
    fprintf(outf, "  TREE sample_%d = [&R] ", s + 1);
    tr_print(outf, tree, /*show_branch_lengths=*/1);
  }
  fprintf(outf, "END;\n\n");
}
