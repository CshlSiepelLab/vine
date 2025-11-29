/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* handling of nuisance parameters in variational inference */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <phast/tree_model.h>
#include <nuisance.h>
#include <nj.h>

/* helper functions for nuisance parameters in variational
   inference. For now these include only the HKY ti/tv parameter for
   DNA models and the silencing rate and leading branch length for
   CRISPR models */
int nj_get_num_nuisance_params(TreeModel *mod, CovarData *data) {
  int retval = 0;

  if (data->crispr_mod != NULL)
    retval += 2;
  else if (mod->subst_mod == HKY85)
    retval += 1;
  else if (mod->subst_mod == REV)
    retval += data->gtr_params->size;

  if (data->rf != NULL)
    retval += data->rf->ctr->size + 2;

  if (data->pf != NULL)
    retval += data->pf->ndim * 2 + 1;

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE)
    retval += (1 + (mod->tree->nnodes + 1)/2 - 1);

  if (data->migtable != NULL)
    retval += data->migtable->gtr_params->size;
  
  return retval;
}

char *nj_get_nuisance_param_name(TreeModel *mod, CovarData *data, int idx) {
  char *tmp;
  assert(idx >= 0);
  if (data->crispr_mod != NULL) {
    if (idx == 0)
      return "nu";
    if (idx == 1)
      return ("lead_t");
    idx -= 2;  /* incrementally subtract each set of indices */
  }
  else if (mod->subst_mod == HKY85) {
    if (idx == 0) 
      return "kappa";
    idx -= 1;
  }
  else if (mod->subst_mod == REV) {
    if (idx < data->gtr_params->size) {
      tmp = smalloc(10 * sizeof(char));
      snprintf(tmp, 10, "gtr[%d]", idx);
      return tmp;
    }
    idx -= data->gtr_params->size;
  }
  
  if (data->rf != NULL) {
    if (idx < data->rf->ctr->size) {
      char *tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "rf_ctr[%d]", idx);
      return tmp;
    }
    idx -= data->rf->ctr->size;
    if (idx == 0)
      return "rf_a";
    if (idx == 1)
      return "rf_b";
    idx -= 2;
  }

  if (data->pf != NULL) {
    if (idx < data->pf->ndim) {
      tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "pf_u[%d]", idx);
      return tmp;
    }
    idx -= data->pf->ndim;
    if (idx < data->pf->ndim) {
      tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "pf_w[%d]", idx);
      return tmp;
    }
    idx -= data->pf->ndim;
    if (idx == 0)
      return "pf_b";
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    if (idx == 0)
      return "relclock_sig";
    idx -= 1;
    if (idx < (mod->tree->nnodes + 1)/2 - 1) {
      tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "nodetime[%d]", idx);
      return tmp;
    }
    idx -= data->treeprior->nodetimes->size;
  }

  if (data->migtable != NULL) {
    if (idx < data->migtable->gtr_params->size) {
      tmp = smalloc(10 * sizeof(char));
      snprintf(tmp, 10, "mig[%d]", idx);
      return tmp;
    }
    idx -= data->migtable->gtr_params->size;
  }
  
  die("ERROR in nj_get_nuisance_param_name: index out of bounds.\n");
  return NULL;
}

/* update nuis_grad based on current gradients */
void nj_update_nuis_grad(TreeModel *mod, CovarData *data, Vector *nuis_grad) {
  int idx = 0, i;
  if (data->crispr_mod != NULL) {
    vec_set(nuis_grad, idx++, data->crispr_mod->deriv_sil);
    vec_set(nuis_grad, idx++, data->crispr_mod->deriv_leading_t);
  }
  else if (mod->subst_mod == HKY85) {
    vec_set(nuis_grad, idx++, data->deriv_hky_kappa);
  }
  else if (mod->subst_mod == REV) {
    for (i = 0; i < data->deriv_gtr->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->deriv_gtr, i));
  }

  if (data->rf != NULL) {
    for (i = 0; i < data->rf->ctr_grad->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->rf->ctr_grad, i));
    vec_set(nuis_grad, idx++, data->rf->a_grad);
    vec_set(nuis_grad, idx++, data->rf->b_grad);
  }

  if (data->pf != NULL) {
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(nuis_grad, idx++, vec_get(data->pf->u_grad, i));
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(nuis_grad, idx++, vec_get(data->pf->w_grad, i));
    vec_set(nuis_grad, idx++, data->pf->b_grad);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    vec_set(nuis_grad, idx++, data->treeprior->relclock_sig_grad);
    for (i = 0; i < data->treeprior->nodetimes_grad->size; i++) 
      vec_set(nuis_grad, idx++, vec_get(data->treeprior->nodetimes_grad, i)); 
  }

  if (data->migtable != NULL) {
    for (i = 0; i < data->migtable->deriv_gtr->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->migtable->deriv_gtr, i));
  }

  assert(idx == nuis_grad->size);
}

/* save current values of nuisance params */
void nj_save_nuis_params(Vector *stored_vals, TreeModel *mod, CovarData *data) {
  int idx = 0, i;
  if (data->crispr_mod != NULL) {
    vec_set(stored_vals, idx++, data->crispr_mod->sil_rate);
    vec_set(stored_vals, idx++, data->crispr_mod->leading_t);
  }
  else if (mod->subst_mod == HKY85) 
    vec_set(stored_vals, idx++, data->hky_kappa);
  else if (mod->subst_mod == REV) {
    for (i = 0; i < data->gtr_params->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->gtr_params, i));
  }
  
  if (data->rf != NULL) {
    for (i = 0; i < data->rf->ctr->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->rf->ctr, i));
    vec_set(stored_vals, idx++, data->rf->a);
    vec_set(stored_vals, idx++, data->rf->b);
  }

  if (data->pf != NULL) {
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(stored_vals, idx++, vec_get(data->pf->u, i));
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(stored_vals, idx++, vec_get(data->pf->w, i));
    vec_set(stored_vals, idx++, data->pf->b);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    vec_set(stored_vals, idx++, data->treeprior->relclock_sig);
    for (i = 0; i < data->treeprior->nodetimes->size; i++) 
      vec_set(stored_vals, idx++, vec_get(data->treeprior->nodetimes, i)); 
  }

  if (data->migtable != NULL) {
    for (i = 0; i < data->migtable->gtr_params->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->migtable->gtr_params, i));
  }
  
  assert(idx == stored_vals->size);
}

/* update all nuisance parameters based on vector of stored values */
void nj_update_nuis_params(Vector *stored_vals, TreeModel *mod, CovarData *data) {
  int idx = 0, i;
  if (data->crispr_mod != NULL) {
    data->crispr_mod->sil_rate = vec_get(stored_vals, idx++);
    data->crispr_mod->leading_t = vec_get(stored_vals, idx++);
  }
  else if (mod->subst_mod == HKY85) {
    data->hky_kappa = vec_get(stored_vals, idx++);
    tm_set_HKY_matrix(mod, data->hky_kappa, -1);
    tm_scale_rate_matrix(mod);
    mm_diagonalize(mod->rate_matrix);
  }
  else if (mod->subst_mod == REV) {
    for (i = 0; i < data->gtr_params->size; i++)
      vec_set(data->gtr_params, i, vec_get(stored_vals, idx++));
    tm_set_rate_matrix(mod, data->gtr_params, 0);
    tm_scale_rate_matrix(mod);
    mm_diagonalize(mod->rate_matrix);
  }
  
  if (data->rf != NULL) {
    if (data->rf->center_update == TRUE)
      for (i = 0; i < data->rf->ctr->size; i++)
        vec_set(data->rf->ctr, i, vec_get(stored_vals, idx++));
    else
      idx += data->rf->ctr->size;
    
    data->rf->a = vec_get(stored_vals, idx++);
    data->rf->b = vec_get(stored_vals, idx++);
    rf_update(data->rf);
  }

  if (data->pf != NULL) {
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(data->pf->u, i, vec_get(stored_vals, idx++)); 
    for (i = 0; i < data->pf->ndim; i++)
      vec_set(data->pf->w, i, vec_get(stored_vals, idx++)); 
    data->pf->b = vec_get(stored_vals, idx++);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    data->treeprior->relclock_sig = vec_get(stored_vals, idx++);
    for (i = 0; i < data->treeprior->nodetimes->size; i++) 
      vec_set(data->treeprior->nodetimes, i, vec_get(stored_vals, idx++));
  }

  if (data->migtable != NULL) {
    for (i = 0; i < data->migtable->gtr_params->size; i++)
      vec_set(data->migtable->gtr_params, i, vec_get(stored_vals, idx++));
    mig_set_REV_matrix(data->migtable, data->migtable->gtr_params);
  }
  
  assert(idx == stored_vals->size);
}

/* add to single nuisance parameter */
void nj_nuis_param_pluseq(TreeModel *mod, CovarData *data, int idx, double inc) {
  if (data->crispr_mod != NULL) {
    if (idx == 0) {
      data->crispr_mod->sil_rate += inc;
      if (data->crispr_mod->sil_rate < 0) 
        data->crispr_mod->sil_rate = 0;
      return;
    }
    if (idx == 1) {
      data->crispr_mod->leading_t += inc;
      if (data->crispr_mod->leading_t < 0) 
        data->crispr_mod->leading_t = 0;
      return;
    }
    idx -= 2; /* subtract for below */
  }
  else if (mod->subst_mod == HKY85) {
    if (idx == 0) {
      data->hky_kappa += inc;
      if (data->hky_kappa < 0)
        data->hky_kappa = 0;
      tm_set_HKY_matrix(mod, data->hky_kappa, -1);
      tm_scale_rate_matrix(mod);
      mm_diagonalize(mod->rate_matrix);
      return;
    }
    idx -= 1;
  }
  else if (mod->subst_mod == REV) {
    if (idx < data->gtr_params->size) {
      vec_set(data->gtr_params, idx, vec_get(data->gtr_params, idx) + inc);
      if (vec_get(data->gtr_params, idx) < 1e-6) vec_set(data->gtr_params, idx, 1e-6);
      tm_set_rate_matrix(mod, data->gtr_params, 0);
      tm_scale_rate_matrix(mod);
      mm_diagonalize(mod->rate_matrix);
      return;
    }
    idx -= data->gtr_params->size;
  }

  if (data->rf != NULL) {
    if (data->rf->center_update == TRUE) {
      if (idx < data->rf->ctr->size) {
        vec_set(data->rf->ctr, idx, vec_get(data->rf->ctr, idx) + inc);
        return;
      }
      idx -= data->rf->ctr->size;
    }
    if (idx == 0) {
      data->rf->a += inc;
      return;
    }
    if (idx == 1) {
      data->rf->b += inc;
      return;
    }
    idx -= 2;
  }

  if (data->pf != NULL) {
    if (idx < data->pf->ndim) {
      vec_set(data->pf->u, idx, vec_get(data->pf->u, idx) + inc);
      return;
    }
    idx -= data->pf->ndim;
    if (idx < data->pf->ndim) {
      vec_set(data->pf->w, idx, vec_get(data->pf->w, idx) + inc);
      return;
    }
    idx -= data->pf->ndim;
    if (idx == 0) {
      data->pf->b += inc;
      return;
    }
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    if (idx == 0) {
      data->treeprior->relclock_sig += inc;
      return;
    }
    idx--;
    if (idx < data->treeprior->nodetimes->size) {
      vec_set(data->treeprior->nodetimes, idx,
              vec_get(data->treeprior->nodetimes, idx) + inc);
      return;
    }
    idx -= data->treeprior->nodetimes->size;
  }

  if (data->migtable) {
    if (idx < data->migtable->gtr_params->size) {
      vec_set(data->migtable->gtr_params, idx, vec_get(data->migtable->gtr_params, idx) + inc);
      if (vec_get(data->migtable->gtr_params, idx) < 1e-6) vec_set(data->migtable->gtr_params, idx, 1e-6);
      mig_set_REV_matrix(data->migtable, data->migtable->gtr_params);
      return;
    }
    idx -= data->migtable->gtr_params->size;
  }

  die("ERROR in nj_nuis_param_pluseq: index out of bounds.\n");
}

/* return value of single nuisance parameter */
double nj_nuis_param_get(TreeModel *mod, CovarData *data, int idx) {
  if (data->crispr_mod != NULL) {
    if (idx == 0)
      return data->crispr_mod->sil_rate;
    if (idx == 1)
      return data->crispr_mod->leading_t;
    idx -= 2; /* subtract for below */
  }
  else if (mod->subst_mod == HKY85) {
    if (idx == 0)
      return data->hky_kappa;
    idx -= 1;
  }
  else if (mod->subst_mod == REV) {
    if (idx < data->gtr_params->size)
      return vec_get(data->gtr_params, idx);
    idx -= data->gtr_params->size;
  }

  if (data->rf != NULL) {
    if (data->rf->center_update == TRUE) {
      if (idx < data->rf->ctr->size) 
        return vec_get(data->rf->ctr, idx);      
      idx -= data->rf->ctr->size;
    }
    if (idx == 0)
      return data->rf->a;
    if (idx == 1)
      return data->rf->b;
    idx -= 2;
  }

  if (data->pf != NULL) {
    if (idx < data->pf->ndim) 
      return vec_get(data->pf->u, idx);
    idx -= data->pf->ndim;
    if (idx < data->pf->ndim) 
      return vec_get(data->pf->w, idx);    
    idx -= data->pf->ndim;
    if (idx == 0)
      return data->pf->b;
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE) {
    if (idx == 0)
      return data->treeprior->relclock_sig;
    idx--;
    if (idx < data->treeprior->nodetimes->size)
      return vec_get(data->treeprior->nodetimes, idx);
    idx -= data->treeprior->nodetimes->size;
  }

  if (data->migtable != NULL) {
    if (idx < data->migtable->gtr_params->size)
      return vec_get(data->migtable->gtr_params, idx);
    idx -= data->migtable->gtr_params->size;
  }
  die("ERROR in nuis_param_get: index out of bounds.\n");
  return -1;
}

