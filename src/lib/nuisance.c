/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
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
#include <phast/dgamma.h>
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

  if (data->dgamma_cats > 1)
    retval += 1;
  
  if (data->rfs[0] != NULL)
    retval += data->rfs[0]->ctr->size + 2;

  if (data->pfs[0] != NULL)
    retval += data->pfs[0]->ndim * 2 + 1;

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE)
    retval += (1 + (mod->tree->nnodes + 1)/2 - 1);

  /* M2 latent clock: relclock_sig + one rate per branch (nnodes-1) */
  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE)
    retval += (1 + (mod->tree->nnodes - 1));

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

  if (data->dgamma_cats > 1) {
    if (idx == 0)
      return "alpha";
    idx -= 1;
  }

  if (data->rfs[0] != NULL) {
    if (idx < data->rfs[0]->ctr->size) {
      char *tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "rf_ctr[%d]", idx);
      return tmp;
    }
    idx -= data->rfs[0]->ctr->size;
    if (idx == 0)
      return "rf_a";
    if (idx == 1)
      return "rf_b";
    idx -= 2;
  }

  if (data->pfs[0] != NULL) {
    if (idx < data->pfs[0]->ndim) {
      tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "pf_u[%d]", idx);
      return tmp;
    }
    idx -= data->pfs[0]->ndim;
    if (idx < data->pfs[0]->ndim) {
      tmp = smalloc(15 * sizeof(char));
      snprintf(tmp, 15, "pf_w[%d]", idx);
      return tmp;
    }
    idx -= data->pfs[0]->ndim;
    if (idx == 0)
      return "pf_b";
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
    if (idx == 0)
      return "relclock_sig";
    idx -= 1;
    /* Use nodetimes->size when allocated (the loop calls
       tp_compute_log_prior at least once, which initializes it).
       The fallback to mod->tree handles the startup header print,
       which happens before any tp_compute_log_prior call.  After
       the variational loop mod->tree has been set to NULL by the
       final nj_elbo_* call, so dereferencing it here would crash. */
    int ninternal = data->treeprior->nodetimes != NULL
                      ? data->treeprior->nodetimes->size
                      : (mod->tree->nnodes + 1)/2 - 1;
    if (idx < ninternal) {
      tmp = smalloc(25 * sizeof(char));
      snprintf(tmp, 25, "nodetime[%d]", idx);
      return tmp;
    }
    idx -= ninternal;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    if (idx == 0)
      return "relclock_sig";
    idx -= 1;
    /* rates->size once allocated (after the first tp_compute_log_prior /
       likelihood pass); tree-based fallback for the startup header print. */
    int nbr = data->treeprior->rates != NULL
                ? data->treeprior->rates->size
                : (mod->tree->nnodes - 1);
    if (idx < nbr) {
      tmp = smalloc(25 * sizeof(char));
      snprintf(tmp, 25, "rate[%d]", idx);
      return tmp;
    }
    idx -= nbr;
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

  if (data->dgamma_cats > 1)
    vec_set(nuis_grad, idx++, data->deriv_dgamma_alpha);
  
  if (data->rfs[0] != NULL) {
    for (i = 0; i < data->rfs[0]->ctr_grad->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->rfs[0]->ctr_grad, i));
    vec_set(nuis_grad, idx++, data->rfs[0]->a_grad);
    vec_set(nuis_grad, idx++, data->rfs[0]->b_grad);
  }

  if (data->pfs[0] != NULL) {
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(nuis_grad, idx++, vec_get(data->pfs[0]->u_grad, i));
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(nuis_grad, idx++, vec_get(data->pfs[0]->w_grad, i));
    vec_set(nuis_grad, idx++, data->pfs[0]->b_grad);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
    vec_set(nuis_grad, idx++, data->treeprior->relclock_sig_grad);
    for (i = 0; i < data->treeprior->nodetimes_grad->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->treeprior->nodetimes_grad, i));
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    vec_set(nuis_grad, idx++, data->treeprior->relclock_sig_grad);
    for (i = 0; i < data->treeprior->rates_grad->size; i++)
      vec_set(nuis_grad, idx++, vec_get(data->treeprior->rates_grad, i));
  }

  if (data->migtable != NULL) {
    /* zero migration gradients during warmup phase */
    double mig_scale = (data->crispr_mod != NULL && data->crispr_mod->mig_warmup) ? 0.0 : 1.0;
    for (i = 0; i < data->migtable->deriv_gtr->size; i++)
      vec_set(nuis_grad, idx++, mig_scale * vec_get(data->migtable->deriv_gtr, i));
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

  if (data->dgamma_cats > 1)
    vec_set(stored_vals, idx++, mod->alpha);
  
  if (data->rfs[0] != NULL) {
    for (i = 0; i < data->rfs[0]->ctr->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->rfs[0]->ctr, i));
    vec_set(stored_vals, idx++, data->rfs[0]->a);
    vec_set(stored_vals, idx++, data->rfs[0]->b);
  }

  if (data->pfs[0] != NULL) {
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(stored_vals, idx++, vec_get(data->pfs[0]->u, i));
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(stored_vals, idx++, vec_get(data->pfs[0]->w, i));
    vec_set(stored_vals, idx++, data->pfs[0]->b);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
    vec_set(stored_vals, idx++, data->treeprior->relclock_sig);
    for (i = 0; i < data->treeprior->nodetimes->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->treeprior->nodetimes, i));
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    vec_set(stored_vals, idx++, data->treeprior->relclock_sig);
    for (i = 0; i < data->treeprior->rates->size; i++)
      vec_set(stored_vals, idx++, vec_get(data->treeprior->rates, i));
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
    if (data->crispr_mod->leading_t < CPR_T_FLOOR)
      data->crispr_mod->leading_t = CPR_T_FLOOR;
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

  if (data->dgamma_cats > 1) {
    mod->alpha = vec_get(stored_vals, idx++);
    DiscreteGamma(mod->freqK, mod->rK, mod->alpha, mod->alpha, 
                  mod->nratecats, 0); 
  }
  
  if (data->rfs[0] != NULL) {
    if (data->rfs[0]->center_update == TRUE)
      for (i = 0; i < data->rfs[0]->ctr->size; i++)
        vec_set(data->rfs[0]->ctr, i, vec_get(stored_vals, idx++));
    else
      idx += data->rfs[0]->ctr->size;
    
    data->rfs[0]->a = vec_get(stored_vals, idx++);
    data->rfs[0]->b = vec_get(stored_vals, idx++);
    rf_update(data->rfs[0]);
  }

  if (data->pfs[0] != NULL) {
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(data->pfs[0]->u, i, vec_get(stored_vals, idx++));
    for (i = 0; i < data->pfs[0]->ndim; i++)
      vec_set(data->pfs[0]->w, i, vec_get(stored_vals, idx++));
    data->pfs[0]->b = vec_get(stored_vals, idx++);
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
    data->treeprior->relclock_sig = vec_get(stored_vals, idx++);
    for (i = 0; i < data->treeprior->nodetimes->size; i++)
      vec_set(data->treeprior->nodetimes, i, vec_get(stored_vals, idx++));
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    data->treeprior->relclock_sig = vec_get(stored_vals, idx++);
    for (i = 0; i < data->treeprior->rates->size; i++)
      vec_set(data->treeprior->rates, i, vec_get(stored_vals, idx++));
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
      else if (data->crispr_mod->sil_rate > CPR_SIL_RATE_MAX)
        data->crispr_mod->sil_rate = CPR_SIL_RATE_MAX;
      return;
    }
    if (idx == 1) {
      data->crispr_mod->leading_t += inc;
      if (data->crispr_mod->leading_t < CPR_T_FLOOR)
        data->crispr_mod->leading_t = CPR_T_FLOOR;
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

  if (data->dgamma_cats > 1) {
    if (idx == 0) {
      mod->alpha += inc;
      if (mod->alpha < 1e-6)
        mod->alpha = 1e-6;
      DiscreteGamma(mod->freqK, mod->rK, mod->alpha, mod->alpha,
                    mod->nratecats, 0);
      return;
    }
    idx -= 1;
  }
  
  if (data->rfs[0] != NULL) {
    if (data->rfs[0]->center_update == TRUE) {
      if (idx < data->rfs[0]->ctr->size) {
        vec_set(data->rfs[0]->ctr, idx, vec_get(data->rfs[0]->ctr, idx) + inc);
        return;
      }
      idx -= data->rfs[0]->ctr->size;
    }
    if (idx == 0) {
      data->rfs[0]->a += inc;
      rf_update(data->rfs[0]);
      return;
    }
    if (idx == 1) {
      data->rfs[0]->b += inc;
      rf_update(data->rfs[0]);
      return;
    }
    idx -= 2;
  }

  if (data->pfs[0] != NULL) {
    if (idx < data->pfs[0]->ndim) {
      vec_set(data->pfs[0]->u, idx, vec_get(data->pfs[0]->u, idx) + inc);
      return;
    }
    idx -= data->pfs[0]->ndim;
    if (idx < data->pfs[0]->ndim) {
      vec_set(data->pfs[0]->w, idx, vec_get(data->pfs[0]->w, idx) + inc);
      return;
    }
    idx -= data->pfs[0]->ndim;
    if (idx == 0) {
      data->pfs[0]->b += inc;
      return;
    }
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
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

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    if (idx == 0) {
      data->treeprior->relclock_sig += inc;
      return;
    }
    idx--;
    if (idx < data->treeprior->rates->size) {
      double v = vec_get(data->treeprior->rates, idx) + inc;
      if (v < 1e-6) v = 1e-6;   /* keep rates positive so bl_eff>0, log finite */
      vec_set(data->treeprior->rates, idx, v);
      return;
    }
    idx -= data->treeprior->rates->size;
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

  if (data->dgamma_cats > 1) {
    if (idx == 0)
      return mod->alpha;
    idx -= 1;
  }
  
  if (data->rfs[0] != NULL) {
    if (data->rfs[0]->center_update == TRUE) {
      if (idx < data->rfs[0]->ctr->size)
        return vec_get(data->rfs[0]->ctr, idx);
      idx -= data->rfs[0]->ctr->size;
    }
    if (idx == 0)
      return data->rfs[0]->a;
    if (idx == 1)
      return data->rfs[0]->b;
    idx -= 2;
  }

  if (data->pfs[0] != NULL) {
    if (idx < data->pfs[0]->ndim)
      return vec_get(data->pfs[0]->u, idx);
    idx -= data->pfs[0]->ndim;
    if (idx < data->pfs[0]->ndim)
      return vec_get(data->pfs[0]->w, idx);
    idx -= data->pfs[0]->ndim;
    if (idx == 0)
      return data->pfs[0]->b;
    idx--;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == FALSE) {
    if (idx == 0)
      return data->treeprior->relclock_sig;
    idx--;
    if (idx < data->treeprior->nodetimes->size)
      return vec_get(data->treeprior->nodetimes, idx);
    idx -= data->treeprior->nodetimes->size;
  }

  if (data->treeprior != NULL && data->treeprior->relclock == TRUE && data->ultrametric == TRUE) {
    if (idx == 0)
      return data->treeprior->relclock_sig;
    idx--;
    if (idx < data->treeprior->rates->size)
      return vec_get(data->treeprior->rates, idx);
    idx -= data->treeprior->rates->size;
  }

  if (data->migtable != NULL) {
    if (idx < data->migtable->gtr_params->size)
      return vec_get(data->migtable->gtr_params, idx);
    idx -= data->migtable->gtr_params->size;
  }
  die("ERROR in nuis_param_get: index out of bounds.\n");
  return -1;
}
