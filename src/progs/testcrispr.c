#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <phast/misc.h>
#include <nj.h>
#include <crispr.h>


int main(int argc, char *argv[]) {
  TreeNode *tree;
  MarkovMatrix *rmat;
  TreeModel *mod;
  double ll;
  CrisprMutTable *newM;
  CrisprMutModel *cprmod;
  Vector *grad;
  FILE *F = phast_fopen(argv[1], "r");
  CrisprMutTable *M = cpr_read_table(F);
  cpr_renumber_states(M);

  newM = cpr_new_sitewise_table(M);

  /* read starting tree */
  tree = tr_new_from_file(phast_fopen(argv[2], "r"));

  /* tree needs to be indexed correctly */
  tr_enforce_unrooted_indexing(tree);

  /* dummy tree model */
  rmat = mm_new(strlen(DEFAULT_ALPHABET), DEFAULT_ALPHABET,
                CONTINUOUS);
  mod = tm_new(tree, rmat, NULL, JC69, DEFAULT_ALPHABET,
               1, 1, NULL, -1);
  tm_set_JC69_matrix(mod);

  cprmod = cpr_new_model(newM, mod, SITEWISE, UNIF);
  cpr_prep_model(cprmod);


  /* compute likelihood and output */
  grad = vec_new(mod->tree->nnodes - 1);
  ll = cpr_compute_log_likelihood(cprmod, grad);

  printf("Log likelihood: %f\n", ll);

  printf("Gradient:\n");
  vec_print(grad, stdout);
    
  cpr_free_table(M);
  cpr_free_model(cprmod);
}
