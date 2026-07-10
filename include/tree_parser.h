/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* shared utilities for reading collections of Newick trees from file,
   used by evalTrees, compareTrees, and any future tree-comparison tools */

#ifndef TREE_PARSER_H
#define TREE_PARSER_H

#include <phast/lists.h>       /* List */
#include <phast/stringsplus.h> /* String */
#include <phast/trees.h>       /* TreeNode */

/* Parse one line containing a Newick tree.  Leading/trailing whitespace and
 * an optional trailing semicolon are removed in place.  Empty lines return
 * NULL. */
TreeNode *tr_parse_newick_line(String *line, const char *fname, int lineno);

/* Read a file of one Newick tree per line (e.g. a posterior sample) into
 * a List of TreeNode*. */
List *tr_read_trees_from_file(const char *fname);

#endif
