/*
 * crisprLnl: evaluate CRISPR log likelihood at a fixed tree and
 * parameters from LAML, for comparison with LAML's reported likelihood.
 *
 * Usage: crisprLnl <mutation_table.tsv> <tree.nwk> <laml_params.txt>
 *
 * The LAML params file should have lines of the form:
 *   Silencing rate: <value>
 *   Mutation rate: <value>
 *   Negative-llh: <value>
 *
 * Branches in the tree are scaled by the LAML mutation rate before
 * likelihood evaluation.  The silencing rate is set from the LAML file.
 * Output: VINE's log likelihood and the negated Negative-llh from LAML.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <phast/misc.h>
#include <phast/trees.h>
#include <nj.h>
#include <crispr.h>

/* parse LAML parameter file; returns 0 on success */
static int parse_laml_params(const char *fname, double *sil_rate,
                             double *mut_rate, double *neg_llh) {
  char line[1024], key[256];
  double val;
  int found_sil = 0, found_mut = 0, found_llh = 0;
  FILE *f = phast_fopen(fname, "r");

  while (fgets(line, sizeof(line), f)) {
    /* split at last colon */
    char *colon = strrchr(line, ':');
    if (colon == NULL) continue;
    *colon = '\0';
    strncpy(key, line, sizeof(key) - 1);
    key[sizeof(key) - 1] = '\0';
    val = atof(colon + 1);

    if (strstr(key, "Silencing rate") != NULL) {
      *sil_rate = val;  found_sil = 1;
    }
    else if (strstr(key, "Mutation rate") != NULL) {
      *mut_rate = val;  found_mut = 1;
    }
    else if (strstr(key, "Negative-llh") != NULL) {
      *neg_llh = val;   found_llh = 1;
    }
  }
  fclose(f);

  if (!found_sil || !found_mut || !found_llh) {
    fprintf(stderr, "ERROR: could not find all required parameters in %s\n",
            fname);
    fprintf(stderr, "  Silencing rate: %s\n", found_sil ? "found" : "MISSING");
    fprintf(stderr, "  Mutation rate:  %s\n", found_mut ? "found" : "MISSING");
    fprintf(stderr, "  Negative-llh:   %s\n", found_llh ? "found" : "MISSING");
    return 1;
  }
  return 0;
}

/* read a Newick tree from a file that may have a [&R] prefix (LAML/BEAST).
   Reads the whole file into a dynamically grown buffer so large trees
   (>100KB) are not silently truncated as they were with the prior
   fixed buffer. */
static TreeNode *read_laml_tree(const char *fname) {
  FILE *f = phast_fopen(fname, "r");
  size_t cap = 16384, len = 0;
  char *buf = smalloc(cap);
  for (;;) {
    if (len == cap - 1) {
      cap *= 2;
      buf = srealloc(buf, cap);
    }
    size_t n = fread(buf + len, 1, cap - 1 - len, f);
    len += n;
    if (n == 0) break;
  }
  fclose(f);
  buf[len] = '\0';

  /* find the first '(' to skip any prefix like [&R] */
  char *p = strchr(buf, '(');
  if (p == NULL)
    die("ERROR: no Newick tree found in %s\n", fname);

  /* strip trailing semicolon and whitespace */
  {
    char *end = p + strlen(p) - 1;
    while (end > p && (*end == ';' || *end == '\n' || *end == '\r' ||
                       *end == ' ' || *end == '\t'))
      *end-- = '\0';
  }

  TreeNode *t = tr_new_from_string(p);
  sfree(buf);
  return t;
}

int main(int argc, char *argv[]) {
  TreeNode *tree;
  MarkovMatrix *rmat;
  TreeModel *mod;
  CrisprMutModel *cprmod;
  Vector *grad;
  double ll, sil_rate, mut_rate, neg_llh, leading_t;

  if (argc != 4) {
    fprintf(stderr,
            "Usage: crisprLnl <mutation_table.tsv> <tree.nwk> <laml_params.txt>\n");
    exit(1);
  }

  /* read LAML parameters */
  if (parse_laml_params(argv[3], &sil_rate, &mut_rate, &neg_llh) != 0)
    exit(1);

  /* fprintf(stderr, "LAML parameters:\n"); */
  /* fprintf(stderr, "  Silencing rate: %.10f\n", sil_rate); */
  /* fprintf(stderr, "  Mutation rate:  %.10f\n", mut_rate); */
  /* fprintf(stderr, "  Negative-llh:   %.10f\n", neg_llh); */

  /* read mutation table */
  FILE *F = phast_fopen(argv[1], "r");
  CrisprMutTable *M = cpr_read_table(F);
  cpr_renumber_states(M);

  /* read starting tree (handles [&R] prefix) */
  tree = read_laml_tree(argv[2]);

  /* scale branches by LAML mutation rate */
  tr_scale(tree, mut_rate);

  /* save the root branch as leading_t before reindexing */
  leading_t = tree->dparent;

  /* tree needs to be indexed correctly */
  tr_enforce_unrooted_indexing(tree);

  /* dummy tree model (subst model is ignored for CRISPR) */
  rmat = mm_new(strlen(DEFAULT_ALPHABET), DEFAULT_ALPHABET, CONTINUOUS);
  mod = tm_new(tree, rmat, NULL, JC69, DEFAULT_ALPHABET, 1, 1, NULL, -1);
  tm_set_JC69_matrix(mod);

  /* create CRISPR model and set parameters from LAML */
  cprmod = cpr_new_model(M, mod, SITEWISE, UNIF);
  cprmod->sil_rate = sil_rate;
  cprmod->leading_t = leading_t;

  cpr_prep_model(cprmod);

  /* compute likelihood */
  grad = vec_new(mod->tree->nnodes - 1);
  ll = cpr_compute_log_likelihood(cprmod, grad);

  /* output comparison */
  printf("VINE log likelihood: %.10f\n", ll);
  printf("LAML log likelihood: %.10f\n", -neg_llh);
  printf("Difference (VINE - LAML): %.10f\n", ll - (-neg_llh));
  printf("Relative difference: %.6f%%\n",
         100.0 * (ll - (-neg_llh)) / (-neg_llh));

  /* clean up */
  vec_free(grad);
  cpr_free_table(M);
  cpr_free_model(cprmod);

  return 0;
}
