#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <phast/misc.h>
#include <nj.h>
#include <phast/sufficient_stats.h>
#include <sparse_matrix.h>
#include <phast/lists.h>

#define KAPPA 4

int main(int argc, char *argv[]) {
  TreeNode *tree;
  MarkovMatrix *rmat;
  TreeModel *mod;
  double ll, lleps;
  Vector *grad;
  CovarData *data;
  Matrix *dmat;
  
  FILE *F = phast_fopen(argv[1], "r");
  msa_format_type format = msa_format_for_content(F, 1);
  MSA * msa = msa_new_from_file_define_format(F, format, DEFAULT_ALPHABET);

  /* Vector *unif = vec_new(4); */
  /* pv_set_uniform(unif); */
  
  if (msa->ss == NULL)
    ss_from_msas(msa, 1, TRUE, NULL, NULL, NULL, -1, 0);

  /* read starting tree */
  tree = tr_new_from_file(phast_fopen(argv[2], "r"));

  /* tree needs to be indexed correctly */
  tr_enforce_unrooted_indexing(tree);

  /* create tree model */
  rmat = mm_new(strlen(DEFAULT_ALPHABET), DEFAULT_ALPHABET,
                CONTINUOUS);
  mod = tm_new(tree, rmat, msa_get_base_freqs(msa, -1, -1), HKY85, DEFAULT_ALPHABET, 
               1, 1, NULL, -1); 
  /* temporary */
  /* mod = tm_new(tree, rmat, unif, HKY85, DEFAULT_ALPHABET,  */
  /*              1, 1, NULL, -1);  */
  tm_set_HKY_matrix(mod, KAPPA, -1);
  tm_scale_rate_matrix(mod);

  /* temporary: JC version */
  /* mod = tm_new(tree, rmat, unif, JC69, DEFAULT_ALPHABET, */
  /*              1, 1, NULL, -1); */
  /* tm_set_JC69_matrix(mod); */
  
  /* set up CovarData */
  dmat = mat_new(5, 5); /* dummy */
  data = nj_new_covar_data(CONST, dmat, 3, msa, NULL, msa->names,
                           FALSE, 1, 3, -1,
                           FALSE, 1, FALSE, FALSE, FALSE, NULL, NULL);
  data->hky_kappa = KAPPA;
    
  /* compute likelihood and output */
  grad = vec_new(mod->tree->nnodes - 1);
  ll = nj_compute_log_likelihood(mod, data, grad);
  printf("Analytical log likelihood: %f\n", ll);

  printf("Analytical gradient:\n"); 
  vec_print(grad, stdout); 

  printf("Analytical kappa gradient: %f\n", data->deriv_hky_kappa);
  
  /* compute numerical gradient for comparison */
  ll = nj_dL_dt_num(grad, mod, data); 
  printf("Numerical log likelihood: %f\n", ll); 
  printf("Numerical gradient:\n"); 
  vec_print(grad, stdout);

  /* compute numerical kappa gradient */
  data->hky_kappa += DERIV_EPS;
  /* tm_print(stdout, mod); */
  tm_set_HKY_matrix(mod, data->hky_kappa, -1);
  tm_scale_rate_matrix(mod);
  mm_diagonalize(mod->rate_matrix);
  /* tm_print(stdout, mod); */
  lleps = nj_compute_log_likelihood(mod, data, NULL);
  printf("Numerical kappa gradient: %f\n", (lleps - ll)/DERIV_EPS);
}
