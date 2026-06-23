/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
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
#include <gradients.h>
#include <phast/sufficient_stats.h>

/* for multithreading */
#ifdef _OPENMP
  #include <omp.h>
  #define NJ_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define NJ_OMP_GET_THREAD_NUM() omp_get_thread_num()
#else
  #define NJ_OMP_GET_MAX_THREADS() 1
  #define NJ_OMP_GET_THREAD_NUM()  0
#endif

/* for use in likelihood calculations to avoid
         underflow */
#define REL_CUTOFF 1e-300

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

/* Compute the log likelihood of a tree model with respect to an
   alignment.  This is a pared-down version of
   tl_compute_log_likelihood from PHAST.  It assumes a 0th order model
   and sufficient stats already available.  It also makes use of
   scaling factors to avoid underflow with large trees.  If gradient
   is non-null, it will be populated with data that can be used to
   compute gradients of the log likelihood.  This "core" routine is
   now threadsafe and is called by wrapper functions that handle
   setup, postprocessing, and multithreading (see below).  The 'range'
   parameter can be used to specify a subset of tuple indices in a
   multithreading setting (set NULL to ignore). */
double nj_ll_core(TreeModel *mod, CovarData *data, NJDerivs *derivs,
                NJGradCache *gcache, List *range) {

  int i, j, k, nodeidx, rcat, tupleidx;
  int nstates = mod->rate_matrix->size;
  TreeNode *n;
  List *traversal;
  double ***pL = NULL, **pLbar = NULL;
  Vector **lscale, *lscale_o; /* inside and outside versions */
  double tmp[nstates];
  MSA *msa = data->msa;
  Vector *this_deriv_gtr = NULL;
  Vector *tuplecounts = gcache->tuplecounts;
  double ll = 0.0;

  double logLk[mod->nratecats];
  double pik[mod->nratecats];
  
  /* allocate inside probabilities, separately per rate category */
  pL = smalloc(mod->nratecats * sizeof(double**));
  for (rcat = 0; rcat < mod->nratecats; rcat++) {
    pL[rcat] = smalloc(nstates * sizeof(double*));
    for (j = 0; j < nstates; j++)
      pL[rcat][j] = smalloc((mod->tree->nnodes+1) * sizeof(double));
  }

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = smalloc(mod->nratecats * sizeof(Vector*));   /* per rate category */
  for (rcat = 0; rcat < mod->nratecats; rcat++)
    lscale[rcat] = vec_new(mod->tree->nnodes+1);
  lscale_o = vec_new(mod->tree->nnodes+1);

  /* initialize derivative accumulators */
  if (derivs != NULL) {
    vec_zero(derivs->branchgrad);
    if (mod->subst_mod == HKY85)
      derivs->deriv_hky_kappa = 0.0;
    else if (mod->subst_mod == REV)
      vec_zero(derivs->deriv_gtr);
    derivs->deriv_dgamma_alpha = 0.0;
  }

  /* for outside algorithm if computing branch gradients */
  if (derivs != NULL) {
    pLbar = smalloc(nstates * sizeof(double*));
    for (j = 0; j < nstates; j++)
      pLbar[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));
    if (mod->subst_mod == REV) {
      this_deriv_gtr = vec_new(data->gtr_params->size);
    }
  }
  
  /* precompute dr_k/dalpha for discrete gamma chain rule */
  Vector *dr_dalpha = NULL;
  if (derivs != NULL && data->dgamma_cats > 1) {
    dr_dalpha = vec_new(mod->nratecats);
    nj_dr_dalpha_gamma(dr_dalpha, mod);
  }

  /* set up active range of tuples for this thread */
  int t0 = 0, t1 = msa->ss->ntuples;
  if (range != NULL) {
    if (lst_size(range) != 2) die("nj_ll_core: range must have size 2");
    t0 = lst_get_int(range, 0);
    t1 = lst_get_int(range, 1);
    if (t0 < 0) t0 = 0;
    if (t1 > msa->ss->ntuples) t1 = msa->ss->ntuples;
  }
  
  for (tupleidx = t0; tupleidx < t1; tupleidx++) {
    if (vec_get(tuplecounts, tupleidx) == 0) continue; /* important when subsampling */

    for (rcat = 0; rcat < mod->nratecats; rcat++) { /* loop over rate categories */

      /* reset scale */
      vec_zero(lscale[rcat]); 
    
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
              pL[rcat][i][n->id] = 1;
            else
              pL[rcat][i][n->id] = 0;
          }
        }
        else {
          /* general recursive case */
          MarkovMatrix *lsubst_mat = mod->P[n->lchild->id][rcat];
          MarkovMatrix *rsubst_mat = mod->P[n->rchild->id][rcat];

          /* do this in a way that handles scaling.  first compute
             unnormalized inside values */
          for (i = 0; i < nstates; i++) {
            double totl = 0.0, totr = 0.0;

            for (j = 0; j < nstates; j++)
              totl += pL[rcat][j][n->lchild->id] *
                      mm_get(lsubst_mat, i, j);

            for (k = 0; k < nstates; k++)
              totr += pL[rcat][k][n->rchild->id] *
                      mm_get(rsubst_mat, i, k);

            pL[rcat][i][n->id] = totl * totr;
          }

          /* nodewise max-normalization across states */
          double maxv = 0.0;
          for (i = 0; i < nstates; i++)
            if (pL[rcat][i][n->id] > maxv)
              maxv = pL[rcat][i][n->id];

          /* propagate scaling from children */
          vec_set(lscale[rcat], n->id,
                  vec_get(lscale[rcat], n->lchild->id) +
                  vec_get(lscale[rcat], n->rchild->id));

          if (maxv > 0.0) {
            /* normalize and update scale */
            for (i = 0; i < nstates; i++)
              pL[rcat][i][n->id] /= maxv;

            vec_set(lscale[rcat], n->id,
                    vec_get(lscale[rcat], n->id) + log(maxv));
          }

          /* zero out tiny values to save time later */
          for (i = 0; i < nstates; i++)
            if (pL[rcat][i][n->id] < REL_CUTOFF)
              pL[rcat][i][n->id] = 0.0;        
        }
      }
  
      /* termination */
      double total_prob = 0;
      for (i = 0; i < nstates; i++)
        total_prob += vec_get(mod->backgd_freqs, i) * pL[rcat][i][mod->tree->id];

      if (total_prob <= 0.0)
        total_prob = REL_CUTOFF;  /* avoid log(0) */

      logLk[rcat] =
        (log(total_prob) + vec_get(lscale[rcat], mod->tree->id));
    } /* end loop over rate categories, inside pass */

    /* compute per-rate-category posteriors, for use in weighting
       likelihoods and derivatives */
    double m = -INFINITY;
    for (rcat = 0; rcat < mod->nratecats; rcat++) 
      if (logLk[rcat] > m) m = logLk[rcat];
    double s = 0.0;
    for (rcat = 0; rcat < mod->nratecats; rcat++)
      s += exp((logLk[rcat]) - m);
    assert(s > 0.0);
    
    double logsum = m + log(s);
    double logmix = logsum - log((double)mod->nratecats);
    /* assumes uniform prior over rate categories; always true now */

    for (rcat = 0; rcat < mod->nratecats; rcat++)
      pik[rcat] = exp(logLk[rcat] - logsum);

    /* accumulate overall log likelihood */
    ll += vec_get(tuplecounts, tupleidx) * logmix;
    
    if (derivs != NULL) {
      for (rcat = 0; rcat < mod->nratecats; rcat++) { /* loop over rate categories */
        /* to compute gradients efficiently, need to make a second pass
           across the tree to compute "outside" probabilities */
        double expon = 0;
        traversal = tr_preorder(mod->tree);

        Vector *this_branchgrad = derivs->branchgrad;

        /* reset scale */
        vec_zero(lscale_o);
        
        for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
          n = lst_get_ptr(traversal, nodeidx);

          if (n->parent == NULL) { /* base case */
            double maxv = 0.0;
            for (i = 0; i < nstates; i++) {
              pLbar[i][n->id] = vec_get(mod->backgd_freqs, i);
              if (pLbar[i][n->id] > maxv)
                maxv = pLbar[i][n->id];
            }

            /* lscale_o[root] is already zero from vec_zero */
            if (maxv > 0.0) {
              for (i = 0; i < nstates; i++)
                pLbar[i][n->id] /= maxv;
              vec_set(lscale_o, n->id, log(maxv));
            }
          }
          else {            /* recursive case */
            TreeNode *sibling = (n == n->parent->lchild ?
                                 n->parent->rchild : n->parent->lchild);
            MarkovMatrix *par_subst_mat = mod->P[n->id][rcat];
            MarkovMatrix *sib_subst_mat = mod->P[sibling->id][rcat];

            /* first form tmp[j] = sum_k pLbar(parent=j) * pL(sibling=k) * P_sib(j,k) */
            for (j = 0; j < nstates; j++) {
              tmp[j] = 0.0;
              double a = pLbar[j][n->parent->id];

              if (a == 0.0) continue;

              for (k = 0; k < nstates; k++) {
                double b = pL[rcat][k][sibling->id];
                if (b > 0.0)
                  tmp[j] += a * b * mm_get(sib_subst_mat, j, k);
              }
            }

            /* now propagate to child */
            for (i = 0; i < nstates; i++) {
              double sum = 0.0;
              for (j = 0; j < nstates; j++)
                sum += tmp[j] * mm_get(par_subst_mat, j, i);
              pLbar[i][n->id] = sum;
            }

            /* nodewise normalization of outside vector */
            double maxv = 0.0;
            for (i = 0; i < nstates; i++)
              if (pLbar[i][n->id] > maxv)
                maxv = pLbar[i][n->id];

            /* bookkeeping for scaling */
            vec_set(lscale_o, n->id,
                    vec_get(lscale_o, n->parent->id) +
                    vec_get(lscale[rcat], sibling->id));

            if (maxv > 0.0) {
              for (i = 0; i < nstates; i++)
                pLbar[i][n->id] /= maxv;

              vec_set(lscale_o, n->id,
                      vec_get(lscale_o, n->id) + log(maxv));
            }

            for (i = 0; i < nstates; i++)
              if (pLbar[i][n->id] < REL_CUTOFF)
                pLbar[i][n->id] = 0.0;
          }
        }

        /* now compute branchwise derivatives in a final pass */
        traversal = mod->tree->nodes;
        for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
          TreeNode *par, *sib;
          double deriv;
          double log_base_prob =
            logLk[rcat] - vec_get(lscale[rcat], mod->tree->id);
          
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
              tmp[i] += pL[rcat][k][sib->id] * mm_get(mod->P[sib->id][rcat], i, k);
          }

          /* compute expon (scaling correction) for every node, including
             rchild, so that the rate-parameter derivatives below use
             the correct scale factor */
          expon = -vec_get(lscale[rcat], mod->tree->id)
                  + vec_get(lscale[rcat], sib->id) + vec_get(lscale_o, par->id)
                  + vec_get(lscale[rcat], n->id) - log_base_prob;
          /* note division by base_prob because we need deriv of log P */

          /* avoid overflow */
          if (expon > 700.0) expon = 700.0;
          if (expon < -745.0) expon = -745.0;

          if (n != mod->tree->rchild) { /* skip branch to right of root because unrooted */
            /* calculate derivative analytically */
            deriv = 0;
            for (i = 0; i < nstates; i++)
              for (j = 0; j < nstates; j++)
                deriv +=  tmp[i] * pLbar[i][par->id] * pL[rcat][j][n->id] * mat_get(gcache->grad_mat[n->id][rcat], i, j);

            deriv *= exp(expon);
            assert(isfinite(deriv));

            vec_set(this_branchgrad, n->id, vec_get(this_branchgrad, n->id) +
                                       deriv * pik[rcat] * vec_get(tuplecounts, tupleidx));

            /* accumulate alpha derivative via chain rule:
               dlogL_k/dalpha = sum_n (dlogL_k/dd_n) * d_n * (dr_k/dalpha) / r_k */
            if (dr_dalpha != NULL && mod->rK[rcat] > 0.0) 
              derivs->deriv_dgamma_alpha +=
                  pik[rcat] * deriv * n->dparent * vec_get(dr_dalpha, rcat) /
                  mod->rK[rcat] * vec_get(tuplecounts, tupleidx);
          }

          /* in these cases, we need a partial derivatives for substitution rates also;
             they have to be aggregated across all branches */
          if (mod->subst_mod == HKY85) {
            double this_deriv_kappa = 0;
            for (i = 0; i < nstates; i++) 
              for (j = 0; j < nstates; j++) 
                this_deriv_kappa += tmp[i] * pLbar[i][par->id] * pL[rcat][j][n->id] *
                                    mat_get(gcache->grad_mat_HKY[n->id][rcat], i, j);

            /* adjust for all relevant scale terms */
            this_deriv_kappa *= exp(expon);
            derivs->deriv_hky_kappa +=
              (this_deriv_kappa * pik[rcat] * vec_get(tuplecounts, tupleidx));
          }
          else if (mod->subst_mod == REV) {
            /* loop over rate parameters */
            for (int pidx = 0; pidx < data->gtr_params->size; pidx++) {
              double pderiv = 0; /* partial deriv wrt this param */
              Matrix *dP_dr = lst_get_ptr(gcache->grad_mat_REV[n->id][rcat], pidx);
              for (int i = 0; i < nstates; i++) {
                for (int j = 0; j < nstates; j++) {
                  pderiv += tmp[i] * pLbar[i][par->id] * pL[rcat][j][n->id] *
                            mat_get(dP_dr, i, j);
                }
              }
              vec_set(this_deriv_gtr, pidx, pderiv);
            }
            /* adjust for all relevant scale terms */
            vec_scale(this_deriv_gtr, exp(expon) * pik[rcat] *
                                      vec_get(tuplecounts, tupleidx));
            vec_plus_eq(derivs->deriv_gtr, this_deriv_gtr);
          }
        } /* end loop over nodes for outside pass */
      }  /* end loop over rate categories for outside pass */
    } /* end if computing derivatives */
  } /* end loop over tuples */

  if (dr_dalpha != NULL) 
    vec_free(dr_dalpha);

  for (rcat = 0; rcat < mod->nratecats; rcat++) {
    for (j = 0; j < nstates; j++) sfree(pL[rcat][j]);
    sfree(pL[rcat]);
    vec_free(lscale[rcat]);
  }
  sfree(pL);
  sfree(lscale);
    
  if (derivs != NULL) {
    for (j = 0; j < nstates; j++)
      sfree(pLbar[j]);
    sfree(pLbar);
    if (mod->subst_mod == REV)
      vec_free(this_deriv_gtr);
  }

  vec_free(lscale_o);

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

/* --- code for multithreading of likelihood calculations --- */

/* The main likelihood function (same interface as previously) is now a wrapper
   that handles setup, subsampling, and multithreading.  The actual likelihood
   calculation is done in nj_ll_core, which is called by
   each thread separately. */
double nj_compute_log_likelihood(TreeModel *mod, CovarData *data,
                                 Vector *branchgrad) {
  double ll;

  /* ---- basic sanity checks  ---- */
 if (data->msa->ss->tuple_size != 1)
    die("ERROR nj_compute_log_likelihood: need tuple size 1, got %i\n",
        data->msa->ss->tuple_size);
  if (mod->order != 0)
    die("ERROR nj_compute_log_likelihood: got mod->order of %i, expected 0\n",
        mod->order);
  if (!mod->allow_gaps)
    die("ERROR nj_compute_log_likelihood: need mod->allow_gaps to be TRUE\n");
    
  /* ---- one-time model setup (not thread-safe if repeated) ---- */
  tm_set_subst_matrices(mod);

  if (mod->msa_seq_idx == NULL)
    tm_build_seq_idx(mod, data->msa);

  tr_postorder(mod->tree); /* ensure these are cached */
  tr_preorder(mod->tree);
  
  /* ---- policy enforcement ---- */
  if (data->nthreads > 1 && data->subsample)
    die("ERROR: subsampling is not allowed with multithreading.\n");

  /* set up tuplecounts; subsample if needed.  Persistent caches
     (CDF and the per-iteration subsample) live on CovarData --
     previously file-scope statics, which broke any consumer with
     more than one MSA in the same process.  The non-subsample path
     hands back a transient borrowed view of msa->ss->counts; the
     existing post-call sfree at the bottom of this function frees
     that view wrapper without touching the MSA-owned data. */
  Vector *tuplecounts;
  if (data->subsample == TRUE) {
    if (data->tuplecdf == NULL) { /* build CDF for sampling */
      Vector *counts = vec_view_array(data->msa->ss->counts, data->msa->ss->ntuples);
      data->tuplecdf = pv_cdf_from_counts(counts, LOWER);
      sfree(counts); /* avoid vec_free (would free the shared underlying array) */
    }
    else  /* CDF already exists; check that it makes sense */
      assert(data->tuplecdf->size == data->msa->ss->ntuples);
    if (data->tuplecounts == NULL || data->reuse_subsamp == FALSE ||
        vec_sum(data->tuplecounts) != data->subsampsize) { /* need new subsample */
      if (data->tuplecounts == NULL)
        data->tuplecounts = vec_new(data->msa->ss->ntuples);
      else
        assert(data->tuplecounts->size == data->msa->ss->ntuples);
      pv_draw_counts(data->tuplecounts, data->tuplecdf, data->subsampsize);
    }
    tuplecounts = data->tuplecounts;
  }
  else {
    tuplecounts = vec_view_array(data->msa->ss->counts,
                                 data->msa->ss->ntuples);
  }

  /* ---- set up gradient cache ---- */
  int nnodes = mod->tree->nnodes;
  int nstates = mod->rate_matrix->size;
  NJGradCache gcache = {0};
  gcache.tuplecounts = tuplecounts;
  if (branchgrad != NULL) {
    int j, p, rcat;
    int ncats = mod->nratecats;

    gcache.grad_mat = malloc(nnodes * sizeof(Matrix **));
    for (j = 0; j < nnodes; j++) {
      gcache.grad_mat[j] = malloc(ncats * sizeof(Matrix *));
      for (rcat = 0; rcat < ncats; rcat++)
        gcache.grad_mat[j][rcat] = mat_new(nstates, nstates);
    }

    if (mod->subst_mod == HKY85) {
      gcache.grad_mat_HKY = malloc(nnodes * sizeof(Matrix **));
      for (j = 0; j < nnodes; j++) {
        gcache.grad_mat_HKY[j] = malloc(ncats * sizeof(Matrix *));
        for (rcat = 0; rcat < ncats; rcat++)
          gcache.grad_mat_HKY[j][rcat] = mat_new(nstates, nstates);
      }
    }
    else if (mod->subst_mod == REV) {
      gcache.grad_mat_REV = malloc(nnodes * sizeof(List **));
      for (j = 0; j < nnodes; j++) {
        gcache.grad_mat_REV[j] = malloc(ncats * sizeof(List *));
        for (rcat = 0; rcat < ncats; rcat++) {
          gcache.grad_mat_REV[j][rcat] = lst_new_ptr(data->gtr_params->size);
          for (p = 0; p < data->gtr_params->size; p++)
            lst_push_ptr(gcache.grad_mat_REV[j][rcat], mat_new(nstates, nstates));
        }
      }
    }

    /* compute gradient matrices ONCE per ratecat */
    for (j = 0; j < nnodes; j++) {
      TreeNode *n = lst_get_ptr(mod->tree->nodes, j);

      for (rcat = 0; rcat < ncats; rcat++) {
        double r = mod->rK[rcat];
        double t = n->dparent * r;

        if (mod->subst_mod == JC69) {
          tm_grad_JC69(mod, gcache.grad_mat[j][rcat], t);
          mat_scale(gcache.grad_mat[j][rcat], r);  /* dP/dd = r * dP/dt */
        }
        else if (mod->subst_mod == HKY85) {
          tm_grad_HKY_dt(mod, gcache.grad_mat[j][rcat], data->hky_kappa, t);
          tm_grad_HKY_dkappa(mod, gcache.grad_mat_HKY[j][rcat], data->hky_kappa, t);
          mat_scale(gcache.grad_mat[j][rcat], r);  /* only dt depends on t */
        }
        else if (mod->subst_mod == REV) {
          tm_grad_REV_dt(mod, gcache.grad_mat[j][rcat], t);
          tm_grad_REV_dr(mod, gcache.grad_mat_REV[j][rcat], t);
          mat_scale(gcache.grad_mat[j][rcat], r);  /* only dt depends on t */
        }
      }
    }
  }

  /* if multiple threads are requested but OpenMP is not active, catch it here */
#ifndef _OPENMP
  if (data->nthreads > 1) 
    die("ERROR: Multithreading requested but OpenMP is not enabled.\n");
#endif
  
  /* ---- sequential path ---- */
  if (data->nthreads == 1) 
    /* legacy behavior, but wrapper-controlled */
    ll = nj_ll_parallel(mod, data, branchgrad, 1, &gcache);

  /* ---- parallel path ---- */
  else
    ll = nj_ll_parallel(mod, data, branchgrad, data->nthreads, &gcache);

  if (data->subsample == FALSE) {
    sfree(tuplecounts);  /* view wrapper */
    tuplecounts = NULL;
  }
    
  if (branchgrad != NULL) {
    int ncats = mod->nratecats;

    for (int j = 0; j < nnodes; j++) {
      for (int rcat = 0; rcat < ncats; rcat++)
        mat_free(gcache.grad_mat[j][rcat]);
      free(gcache.grad_mat[j]);
    }
    free(gcache.grad_mat);

    if (gcache.grad_mat_HKY) {
      for (int j = 0; j < nnodes; j++) {
        for (int rcat = 0; rcat < ncats; rcat++)
          mat_free(gcache.grad_mat_HKY[j][rcat]);
        free(gcache.grad_mat_HKY[j]);
      }
      free(gcache.grad_mat_HKY);
    }

    if (gcache.grad_mat_REV) {
      for (int j = 0; j < nnodes; j++) {
        for (int rcat = 0; rcat < ncats; rcat++) {
          List *gm = gcache.grad_mat_REV[j][rcat];
          for (int p = 0; p < lst_size(gm); p++)
            mat_free(lst_get_ptr(gm, p));
          lst_free(gm);
        }
        free(gcache.grad_mat_REV[j]);
      }
      free(gcache.grad_mat_REV);
    }
  }

  return ll;
}

double nj_ll_parallel(TreeModel *mod, CovarData *data, Vector *branchgrad,
                      int nthreads_requested, NJGradCache *gcache) {

  int ntuples = data->msa->ss->ntuples;

  int maxthreads = NJ_OMP_GET_MAX_THREADS();
  int nthreads = (nthreads_requested <= maxthreads ? nthreads_requested : maxthreads);

  double ll_total = 0.0;

  /* ---- allocate per-thread accumulators ---- */
  NJDerivs **thread_derivs = NULL;
  double *thread_ll = calloc(nthreads, sizeof(double));

  if (branchgrad != NULL) {
    thread_derivs = malloc(nthreads * sizeof(NJDerivs *));

    for (int t = 0; t < nthreads; t++) {
      thread_derivs[t] = malloc(sizeof(NJDerivs));
      thread_derivs[t]->branchgrad = vec_new(branchgrad->size);
      vec_zero(thread_derivs[t]->branchgrad);

      thread_derivs[t]->deriv_hky_kappa = 0.0;
      thread_derivs[t]->deriv_dgamma_alpha = 0.0;
      thread_derivs[t]->deriv_gtr =
        (mod->subst_mod == REV ? vec_new(data->gtr_params->size) : NULL);
    }
  }
  
  /* ---- parallel likelihood computation ---- */
#pragma omp parallel num_threads(nthreads)
  {
    int tid = NJ_OMP_GET_THREAD_NUM();

    int start = (tid * ntuples) / nthreads;
    int end = ((tid + 1) * ntuples) / nthreads;

    List *range = lst_new_int(2);
    lst_push_int(range, start);
    lst_push_int(range, end);

    thread_ll[tid] = nj_ll_core(mod, data, (branchgrad ? thread_derivs[tid] : NULL),
               gcache, range);
    
    lst_free(range);
  }

  /* ---- reduction ---- */
  for (int t = 0; t < nthreads; t++)
    ll_total += thread_ll[t];

  if (branchgrad != NULL) {
    vec_zero(branchgrad);
    for (int t = 0; t < nthreads; t++)
      vec_plus_eq(branchgrad, thread_derivs[t]->branchgrad);

    if (mod->subst_mod == HKY85) {
      data->deriv_hky_kappa = 0.0;
      for (int t = 0; t < nthreads; t++)
        data->deriv_hky_kappa += thread_derivs[t]->deriv_hky_kappa;
    }

    if (mod->subst_mod == REV) {
      vec_zero(data->deriv_gtr);
      for (int t = 0; t < nthreads; t++)
        vec_plus_eq(data->deriv_gtr, thread_derivs[t]->deriv_gtr);
    }

    if (data->dgamma_cats > 1) {
      data->deriv_dgamma_alpha = 0.0;
      for (int t = 0; t < nthreads; t++)
        data->deriv_dgamma_alpha += thread_derivs[t]->deriv_dgamma_alpha;
    }
  }
  
  /* ---- cleanup ---- */
  free(thread_ll);

  if (branchgrad != NULL) {
    for (int t = 0; t < nthreads; t++) {
      vec_free(thread_derivs[t]->branchgrad);

      if (thread_derivs[t]->deriv_gtr) 
        vec_free(thread_derivs[t]->deriv_gtr);

      free(thread_derivs[t]);
    }
    free(thread_derivs);
  }  

  return ll_total;
}
