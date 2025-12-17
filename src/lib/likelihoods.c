/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/*
 * Portions of this file are derived from the PHAST package,
 * distributed under the BSD 3-Clause License.
 */

/* phylogenetic likelihood calculations for DNA models */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>
#include <likelihoods.h>
#include <phast/sufficient_stats.h>

/* fully reset a tree model for use in likelihood calculations with a
   new tree */
void nj_reset_tree_model(TreeModel *mod, TreeNode *newtree) {
  if (mod->tree != NULL)
    tr_free(mod->tree);

  mod->tree = newtree;
  
  /* the size of the tree won't change, so we don't need to do a full
     call to tm_reset_tree; we also don't need to call
     tm_set_subst_matrices because it will be called each time the
     likelihood function is invoked */
}

/* Compute and return the log likelihood of a tree model with respect
   to an alignment.  This is a pared-down version of
   tl_compute_log_likelihood from PHAST.  It assumes a 0th order
   model, sufficient stats already available, and no rate variation.
   It also makes use of scaling factors to avoid underflow with large
   trees.  If branchgrad is non-null, it will be populated with the
   gradient of the log likelihood with respect to the individual
   branches of the tree, in post-order.  */
double nj_compute_log_likelihood(TreeModel *mod, CovarData *data, Vector *branchgrad) {

  int i, j, k, nodeidx, rcat = 0, tupleidx;
  int nstates = mod->rate_matrix->size;
  TreeNode *n;
  double total_prob = 0;
  List *traversal;
  double **pL = NULL, **pLbar = NULL;
  Vector *lscale, *lscale_o; /* inside and outside versions */
  double scaling_threshold = sqrt(DBL_MIN) * 1.0e15;  /* need some padding */
  double lscaling_threshold = log(scaling_threshold), ll = 0;
  double tmp[nstates];
  Matrix **grad_mat, **grad_mat_HKY;
  List **grad_mat_REV;
  unsigned int rescale;
  MSA *msa = data->msa;
  Vector *this_deriv_gtr = NULL;
  static Vector *tuplecdf = NULL;
  static Vector *tuplecounts = NULL;

  if (msa->ss->tuple_size != 1)
    die("ERROR nj_compute_log_likelihood: need tuple size 1, got %i\n",
	msa->ss->tuple_size);
  if (mod->order != 0)
    die("ERROR nj_compute_log_likelihood: got mod->order of %i, expected 0\n",
	mod->order);
  if (!mod->allow_gaps)
    die("ERROR nj_compute_log_likelihood: need mod->allow_gaps to be TRUE\n");
  if (mod->nratecats > 1)
    die("ERROR nj_compute_log_likelihood: no rate variation allowed\n");
  
  pL = smalloc(nstates * sizeof(double*));
  for (j = 0; j < nstates; j++)
    pL[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = vec_new(mod->tree->nnodes+1); 
  lscale_o = vec_new(mod->tree->nnodes+1);
  
  if (branchgrad != NULL) {
    if (branchgrad->size != mod->tree->nnodes-1) /* rooted */
      die("ERROR in nj_compute_log_likelihood: size of branchgrad must be 2n-2\n");
    vec_zero(branchgrad);
    pLbar = smalloc(nstates * sizeof(double*));
    for (j = 0; j < nstates; j++)
      pLbar[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));
    if (mod->subst_mod == HKY85)
      data->deriv_hky_kappa = 0.0;
    else if (mod->subst_mod == REV)
      vec_zero(data->deriv_gtr);
    grad_mat = malloc(mod->tree->nnodes * sizeof(void*));
    for (j = 0; j < mod->tree->nnodes; j++)
      grad_mat[j] = mat_new(nstates, nstates);
    if (mod->subst_mod == HKY85) {
      grad_mat_HKY = malloc(mod->tree->nnodes * sizeof(void*));
      for (j = 0; j < mod->tree->nnodes; j++)
        grad_mat_HKY[j] = mat_new(nstates, nstates);
    }
    else if (mod->subst_mod == REV) {
      /* in this case, each node of the tree needs a list of gradient
         matrices, one for each free GTR parameter */
      grad_mat_REV = malloc(mod->tree->nnodes * sizeof(void*));
      for (j = 0; j < mod->tree->nnodes; j++) {
        grad_mat_REV[j] = lst_new_ptr(data->gtr_params->size);
        for (int jj = 0; jj < data->gtr_params->size; jj++)
          lst_push_ptr(grad_mat_REV[j],
                       mat_new(nstates, nstates));
      }
      this_deriv_gtr = vec_new(data->gtr_params->size);
    }
  }

  tm_set_subst_matrices(mod);  /* just call this in all cases; we'll be tweaking the model a lot */
  
  /* get sequence index if not already there */
  if (mod->msa_seq_idx == NULL)
    tm_build_seq_idx(mod, msa);

  /* set up tuple counts, subsampling if necessary */
  if (data->subsample == TRUE) {
    if (tuplecdf == NULL) { /* build CDF for sampling */
      Vector *counts = vec_view_array(msa->ss->counts, msa->ss->ntuples);
      tuplecdf = pv_cdf_from_counts(counts, LOWER);
      sfree(counts); /* avoid vec_free */
    }
    else /* CDF already exists; check that it makes sense */
      assert(tuplecdf->size == msa->ss->ntuples);
    if (tuplecounts == NULL || data->reuse_subsamp == FALSE ||
        vec_sum(tuplecounts) != data->subsampsize) { /* need new subsample */
      if (tuplecounts == NULL) tuplecounts = vec_new(msa->ss->ntuples);
      else assert(tuplecounts->size == msa->ss->ntuples);
      pv_draw_counts(tuplecounts, tuplecdf, data->subsampsize);
    }
  }
  else /* not subsampling; just provide a 'view' of the full counts array */
    tuplecounts = vec_view_array(msa->ss->counts, msa->ss->ntuples);
      
  for (tupleidx = 0; tupleidx < msa->ss->ntuples; tupleidx++) {
    if (vec_get(tuplecounts, tupleidx) == 0) continue; /* important when subsampling */

    /* reset scale */
    vec_zero(lscale); vec_zero(lscale_o);
    
    traversal = tr_postorder(mod->tree);
    for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
      n = lst_get_ptr(traversal, nodeidx);

      if (n->lchild == NULL) {
        /* leaf: base case of recursion */
        int thisseq = mod->msa_seq_idx[n->id];
        
        if (thisseq < 0) 
          die("No match in alignment for leaf %s.\n", n->name);
        
        int state = mod->rate_matrix->
          inv_states[(int)ss_get_char_tuple(msa, tupleidx,
                                            thisseq, 0)];

        for (i = 0; i < nstates; i++) {
          if (state < 0 || i == state)
            pL[i][n->id] = 1;
          else
            pL[i][n->id] = 0;
        }
      }
      else {
        /* general recursive case */
        MarkovMatrix *lsubst_mat = mod->P[n->lchild->id][rcat];
        MarkovMatrix *rsubst_mat = mod->P[n->rchild->id][rcat];

        /* set values in one or two passes, depending on whether
           rescaling is needed */
        rescale = FALSE;
        for (int pass = 0; pass == 0 || rescale; pass++) {
          for (i = 0; i < nstates; i++) {
            double totl = 0.0, totr = 0.0;
            for (j = 0; j < nstates; j++)
              totl += pL[j][n->lchild->id] *
                mm_get(lsubst_mat, i, j);
          
            for (k = 0; k < nstates; k++)
              totr += pL[k][n->rchild->id] *
                mm_get(rsubst_mat, i, k);

            if (pass == 0 && totl > 0.0 && totr > 0.0 &&
                (totl < scaling_threshold || totr < scaling_threshold)) 
              rescale = TRUE; /* will trigger second pass */
            
            if (pass == 1)  /* second pass: do rescaling */
              pL[i][n->id] = (totl / scaling_threshold) * (totr / scaling_threshold); 
            else
              pL[i][n->id] = totl * totr;
          }
        }

        /* deal with nodewise scaling */
        vec_set(lscale, n->id, vec_get(lscale, n->lchild->id) +
                vec_get(lscale, n->rchild->id));
        if (rescale == TRUE)  /* have to rescale for all states */
          vec_set(lscale, n->id, vec_get(lscale, n->id) + 2 * lscaling_threshold);
        
        /* TEMPORARY: check nonzero */
        double checksum = 0.0;
        for (i = 0; i < nstates; i++)
          checksum += pL[i][n->id];
        assert(checksum > 0.0);
      }
    }
  
    /* termination */
    total_prob = 0;
    for (i = 0; i < nstates; i++)
      total_prob += vec_get(mod->backgd_freqs, i) *
        pL[i][mod->tree->id] * mod->freqK[rcat];      
    
    ll += (log(total_prob) + vec_get(lscale, mod->tree->id)) * vec_get(tuplecounts, tupleidx);

    if (!isfinite(ll))
      assert(0);
      /* break; */ /* can happen with zero-length branches;
                                 make calling code deal with it */

    /* to compute gradients efficiently, need to make a second pass
       across the tree to compute "outside" probabilities */
    if (branchgrad != NULL) {
      double expon;
      traversal = tr_preorder(mod->tree);

      for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
        n = lst_get_ptr(traversal, nodeidx);

        if (n->parent == NULL) { /* base case */
          for (i = 0; i < nstates; i++)
            pLbar[i][n->id] = vec_get(mod->backgd_freqs, i);
        }
        else {            /* recursive case */
          TreeNode *sibling = (n == n->parent->lchild ?
                               n->parent->rchild : n->parent->lchild);
          MarkovMatrix *par_subst_mat = mod->P[n->id][rcat];
          MarkovMatrix *sib_subst_mat = mod->P[sibling->id][rcat];

          rescale = FALSE;

          /* set values in one or two passes, depending on whether rescaling is needed */
          for (int pass = 0; pass == 0 || rescale; pass++) {
    
            for (j = 0; j < nstates; j++) { /* parent state */
              tmp[j] = 0.0;
              for (k = 0; k < nstates; k++) {  /* sibling state */
                double a = pLbar[j][n->parent->id];
                double b = pL[k][sibling->id];

                if (pass == 0 && a > 0.0 && b > 0.0 &&
                    (a < scaling_threshold || b < scaling_threshold))
                  rescale = TRUE;

                if (pass == 1)  /* safe: divide by t twice */
                  tmp[j] += (a / scaling_threshold) * (b / scaling_threshold) *
                    mm_get(sib_subst_mat, j, k);
                else
                  tmp[j] += a * b * mm_get(sib_subst_mat, j, k);
              }
            }
          
            for (i = 0; i < nstates; i++) { /* child state */
              pLbar[i][n->id] = 0.0;
              for (j = 0; j < nstates; j++) { /* parent state */
                pLbar[i][n->id] +=
                  tmp[j] * mm_get(par_subst_mat, j, i);
              }
            }
          } /* end passes */
          
          vec_set(lscale_o, n->id, vec_get(lscale_o, n->parent->id) +
                  vec_get(lscale, sibling->id));
          if (rescale == TRUE) { /* rescale for all states */
            vec_set(lscale_o, n->id, vec_get(lscale_o, n->id) + 2*lscaling_threshold);
          }
        }
      }

      /* TEMPORARY: check inside/outside */
      /* for (nodeidx = 0; nodeidx < lst_size(mod->tree->nodes); nodeidx++) { */
      /*   double pr = 0; */
      /*   n = lst_get_ptr(mod->tree->nodes, nodeidx); */
      /*   assert(vec_get(lscale, n->id) == 0 && vec_get(lscale_o, n->id) == 0); */
      /*   for (j = 0; j < nstates; j++) */
      /*     pr += pL[j][n->id] * pLbar[j][n->id]; */
      /*   printf("Tuple %d, node %d: %f (%f)\n", tupleidx, nodeidx, log(pr), log(total_prob)); */
      /* } */

      
      /* now compute branchwise derivatives in a final pass */
      traversal = mod->tree->nodes;
      for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
        TreeNode *par, *sib;
        double base_prob = total_prob, deriv;
        
        n = lst_get_ptr(traversal, nodeidx);
        par = n->parent;
	
        if (par == NULL) 
          continue;
       
        sib = (n == n->parent->lchild ?
               n->parent->rchild : n->parent->lchild);

        /* this part is just a constant to propagate through to the
           derivative */
        for (i = 0; i < nstates; i++) {  /* parent */
          tmp[i] = 0;
          for (k = 0; k < nstates; k++)  /* sibling */
            tmp[i] += pL[k][sib->id] * mm_get(mod->P[sib->id][rcat], i, k);
        }

        if (n != mod->tree->rchild) { /* skip branch to right of root because unrooted */
          /* calculate derivative analytically */
          deriv = 0;
          /* only do this first time through */
          if (tupleidx == 0) {
            if (mod->subst_mod == JC69)
              tm_grad_JC69(mod, grad_mat[n->id], n->dparent);
            else if (mod->subst_mod == HKY85)
              tm_grad_HKY_dt(mod, grad_mat[n->id], data->hky_kappa, n->dparent);
            else if (mod->subst_mod == REV) 
              tm_grad_REV_dt(mod, grad_mat[n->id], n->dparent); 
            else
              die("ERROR in nj_compute_log_likelihood: only JC69, HKY85 and REV substitution models are supported.\n");
          }
          
          for (i = 0; i < nstates; i++)   
            for (j = 0; j < nstates; j++)    
              deriv +=  tmp[i] * pLbar[i][par->id] * pL[j][n->id] * mat_get(grad_mat[n->id], i, j);

          /* adjust for all relevant scale terms; do everything in log space */
          expon = -vec_get(lscale, mod->tree->id)
            + vec_get(lscale, sib->id) + vec_get(lscale_o, par->id)
            + vec_get(lscale, n->id) - log(base_prob);
          /* note division by base_prob because we need deriv of log P */

          /* avoid overflow */
          if (expon > 700.0) expon = 700.0;
          if (expon < -745.0) expon = -745.0;
          
          deriv *= exp(expon);
          assert(isfinite(deriv));
                  
          vec_set(branchgrad, n->id, vec_get(branchgrad, n->id) +
                  deriv * vec_get(tuplecounts, tupleidx));
        }

        /* in these cases, we need a partial derivatives for substitution rates also;
           they have to be aggregated across all branches */
        if (mod->subst_mod == HKY85) {
          double this_deriv_kappa = 0;
          if (tupleidx == 0)
            tm_grad_HKY_dkappa(mod, grad_mat_HKY[n->id], data->hky_kappa, n->dparent);
          for (i = 0; i < nstates; i++) 
            for (j = 0; j < nstates; j++) 
              this_deriv_kappa += tmp[i] * pLbar[i][par->id] * pL[j][n->id] *
                mat_get(grad_mat_HKY[n->id], i, j);

          /* adjust for all relevant scale terms */
          this_deriv_kappa *= exp(expon);        
          data->deriv_hky_kappa += (this_deriv_kappa * vec_get(tuplecounts, tupleidx));
        }
        else if (mod->subst_mod == REV) {
          if (tupleidx == 0) /* first time only */
            tm_grad_REV_dr(mod, grad_mat_REV[n->id], n->dparent);
          /* loop over rate parameters */
          for (int pidx = 0; pidx < data->gtr_params->size; pidx++) {
            double pderiv = 0; /* partial deriv wrt this param */
            Matrix *dP_dr = lst_get_ptr(grad_mat_REV[n->id], pidx);
            for (int i = 0; i < nstates; i++) {
              for (int j = 0; j < nstates; j++) {
                pderiv += tmp[i] * pLbar[i][par->id] * pL[j][n->id] *
                  mat_get(dP_dr, i, j);
              }
            }
            vec_set(this_deriv_gtr, pidx, pderiv);
          }
          /* adjust for all relevant scale terms */
          vec_scale(this_deriv_gtr, exp(expon));
          vec_plus_eq(data->deriv_gtr, this_deriv_gtr);
        }
      }
    }
  }
  
  for (j = 0; j < nstates; j++)
    sfree(pL[j]);
  sfree(pL);
  
  if (branchgrad != NULL) {
    for (j = 0; j < nstates; j++)
      sfree(pLbar[j]);
    sfree(pLbar);
    for (j = 0; j < mod->tree->nnodes; j++)      
      mat_free(grad_mat[j]);
    free(grad_mat);
    if (mod->subst_mod == HKY85) {
      for (j = 0; j < mod->tree->nnodes; j++)      
        mat_free(grad_mat_HKY[j]);
      free(grad_mat_HKY);
    }
    else if (mod->subst_mod == REV) {
      for (j = 0; j < mod->tree->nnodes; j++) {
        List *gmats = grad_mat_REV[j];
        for (int jj = 0; jj < lst_size(gmats); jj++)
          mat_free(lst_get_ptr(gmats, jj));
        lst_free(gmats);
      }
      free(grad_mat_REV);
      vec_free(this_deriv_gtr);
    }
  }

  vec_free(lscale);
  vec_free(lscale_o);

  if (data->subsample == FALSE) {
    sfree(tuplecounts); /* in this case, it's just a wrapper for the
                           underlying array of counts; avoid vec_free;
                           don't do anything if subsampling because
                           have to store for possible reuse */
    tuplecounts = NULL;
  }
  
  return ll;
}

/* Build index of leaf ids to sequence indices based on a name
   list. */
int *nj_build_seq_idx(List *leaves, char **names) {
  int i;  
  int *retval = smalloc(lst_size(leaves)*2 * sizeof(int));
  for (i = 0; i < lst_size(leaves)*2; i++) retval[i] = -1;
  for (i = 0; i < lst_size(leaves); i++) {
    TreeNode *n = lst_get_ptr(leaves, i);
    retval[n->id] = nj_get_seq_idx(names, n->name, lst_size(leaves));
    if (retval[n->id] < 0)
      die("ERROR: leaf '%s' not found in name list.\n", n->name);
  }
  return retval;
}

/* Return index of given sequence name or -1 if not found. */
int nj_get_seq_idx(char **names, char *name, int n) {
  int i, retval = -1;
  for (i = 0; retval < 0 && i < n; i++) 
    if (!strcmp(name, names[i]))
      retval = i;
  return retval;
}

/* custom setup for parameter mapping for GTR parameterization.  Set
   up lists that allow each rate matrix parameter to be mapped to the
   rows and columns in which it appears; this is a simplified version
   of tm_init_rmp */
void nj_init_gtr_mapping(TreeModel *tm) {
  tm->rate_matrix_param_row = (List**)smalloc(GTR_NPARAMS * sizeof(List*));
  tm->rate_matrix_param_col = (List**)smalloc(GTR_NPARAMS * sizeof(List*));
  for (int i = 0; i < GTR_NPARAMS; i++) {
    tm->rate_matrix_param_row[i] = lst_new_int(2);
    tm->rate_matrix_param_col[i] = lst_new_int(2);
  }
}
