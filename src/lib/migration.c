/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* handling of migration models for phylogeny reconstruction */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <phast/misc.h>
#include <phast/stringsplus.h>
#include <phast/hashtable.h>
#include <crispr.h>
#include <migration.h>
#include <multiDAG.h>

/* for use in likelihood calculations to avoid underflow */
#define REL_CUTOFF 1e-300 

MigTable *mig_new() {
  MigTable *M = smalloc(sizeof(MigTable));
  M->cellnames = lst_new_ptr(200);
  M->states = lst_new_int(50);
  M->statehash = hsh_new(50);
  M->statenames = lst_new_ptr(50);
  M->ncells = 0;
  M->nstates = 0;
  M->nparams = 0;
  M->gtr_params = NULL;
  M->deriv_gtr = NULL;
  M->Pt = NULL;
  M->backgd_freqs = NULL;
  M->rate_matrix = NULL;
  M->rate_matrix_param_row = NULL;
  M->rate_matrix_param_col = NULL;
  M->primary_state = -1;
  return M;
}

void mig_free(MigTable *M) {
  lst_free_strings(M->cellnames);
  lst_free(M->cellnames);
  lst_free(M->states);
  hsh_free(M->statehash);
  lst_free_strings(M->statenames);
  lst_free(M->statenames);
  sfree(M);
}

/* dump migration table to file in same format as read, plus state enumeration */
void mig_dump_table(MigTable *M, FILE *F) {
  int i;

  fprintf(F, "# Migration table dump: ncells=%d, nstates=%d, nparams=%d, primary_state=%d\n",
          M->ncells, M->nstates, M->nparams, M->primary_state);

  /* output cell-state mappings in original format */
  for (i = 0; i < M->ncells; i++) {
    String *cellname = lst_get_ptr(M->cellnames, i);
    int stateidx = lst_get_int(M->states, i);
    String *statename = lst_get_ptr(M->statenames, stateidx);
    fprintf(F, "%s,%s\n", cellname->chars, statename->chars);
  }

  /* enumerate all states with their indices */
  fprintf(F, "# State enumeration:\n");
  for (i = 0; i < M->nstates; i++) {
    String *statename = lst_get_ptr(M->statenames, i);
    /* verify hash table consistency */
    int hashidx = hsh_get_int(M->statehash, statename->chars);
    fprintf(F, "#   state[%d] = \"%s\" (hash lookup returns %d)%s\n",
            i, statename->chars, hashidx,
            (hashidx != i) ? " *** MISMATCH ***" : "");
  }
  fprintf(F, "# lst_size(statenames)=%d, lst_size(states)=%d, lst_size(cellnames)=%d\n",
          lst_size(M->statenames), lst_size(M->states), lst_size(M->cellnames));
}

/* read migration table from file; expects two comma-delimited
   columns: cell name, state name */
MigTable *mig_read_table(FILE *F) {
  MigTable *M = mig_new();
  String *line = str_new(STR_MED_LEN);
  List *cols = lst_new_ptr(2);
  int lineno = 0;
  while (str_readline(line, F) != EOF) {
    lineno++;

    if (str_starts_with_charstr(line, "#")) /* comment line */
      continue;

    str_split(line, ",", cols);
    if (lst_size(cols) != 2)
      die("ERROR in line %d of input file: each line must have two columns (comma-delimited)\n", lineno);
    lst_push_ptr(M->cellnames, lst_get_ptr(cols, 0));

    String *statename = lst_get_ptr(cols, 1);
    str_double_trim(statename);
    int stateno = hsh_get_int(M->statehash, statename->chars);
    if (stateno == -1) { /* new state */
      stateno = M->nstates++;
      hsh_put_int(M->statehash, statename->chars, stateno);
      lst_push_ptr(M->statenames, str_dup(statename));      
    }
    
    lst_push_int(M->states, stateno);
  }

  lst_free(cols);
  str_free(line);

  M->ncells = lst_size(M->cellnames);
  mig_update_states(M);
  return M;
}

/* set primary state by label.  If the label is not found in the
   migration table, the state space is expanded to include it.
   Returns 0 on success. */
int mig_set_primary_state(MigTable *M, const char *statelabel) {
  int idx = hsh_get_int(M->statehash, statelabel);
  if (idx == -1) {
    idx = M->nstates++;
    hsh_put_int(M->statehash, statelabel, idx);
    lst_push_ptr(M->statenames, str_new_charstr(statelabel));
    /* reinitialize rate matrix and parameters for expanded state space */
    mig_update_states(M);
  }
  M->primary_state = idx;
  return 0;
}

void mig_update_states(MigTable *M) {
  int i;

  /* free existing allocations if present (allows re-calling after state expansion) */
  if (M->gtr_params != NULL) vec_free(M->gtr_params);
  if (M->deriv_gtr != NULL) vec_free(M->deriv_gtr);
  if (M->backgd_freqs != NULL) vec_free(M->backgd_freqs);
  if (M->rate_matrix_param_row != NULL) {
    for (i = 0; i < M->nparams; i++)
      if (M->rate_matrix_param_row[i] != NULL) lst_free(M->rate_matrix_param_row[i]);
    sfree(M->rate_matrix_param_row);
  }
  if (M->rate_matrix_param_col != NULL) {
    for (i = 0; i < M->nparams; i++)
      if (M->rate_matrix_param_col[i] != NULL) lst_free(M->rate_matrix_param_col[i]);
    sfree(M->rate_matrix_param_col);
  }
  if (M->rate_matrix != NULL) mm_free(M->rate_matrix);
  if (M->Pt != NULL) {
    for (i = 0; i < lst_size(M->Pt); i++)
      mm_free(lst_get_ptr(M->Pt, i));
    lst_free(M->Pt);
    M->Pt = NULL;
  }

  M->nparams = (M->nstates * (M->nstates - 1)) / 2;
  M->gtr_params = vec_new(M->nparams);
  vec_set_random(M->gtr_params, 1.0, 0.1);
  M->deriv_gtr = vec_new(M->nparams);
  vec_zero(M->deriv_gtr);
  M->backgd_freqs = vec_new(M->nstates); /* has to be done before below */
  vec_set_all(M->backgd_freqs, 1.0 / M->nstates);
  M->rate_matrix_param_row = (List**)smalloc(M->nparams * sizeof(List*));
  M->rate_matrix_param_col = (List**)smalloc(M->nparams * sizeof(List*));
  for (i = 0; i < M->nparams; i++) {
    M->rate_matrix_param_row[i] = lst_new_int(2);
    M->rate_matrix_param_col[i] = lst_new_int(2);
  }
  M->rate_matrix = mm_new(M->nstates, NULL, DISCRETE);
  mm_set_eigentype(M->rate_matrix, REAL_NUM);
  mig_set_REV_matrix(M, M->gtr_params);
}

/* check that migration table contains the same list of cellnames as a
   mutation table. Also sort migration table cells to match mutation
   table */
void mig_check_table(MigTable *mg, CrisprMutTable *mm) {
  int i;
  if (mg->ncells != mm->ncells)
    die("ERROR: migration table and mutation table have different numbers of "
        "cells.\n");

  /* build hashtable for migration table cellnames */
  Hashtable *mignamehash = hsh_new(mg->ncells * 2);
  for (i = 0; i < mg->ncells; i++) {
    String *s = lst_get_ptr(mg->cellnames, i);
    hsh_put_int(mignamehash, s->chars, i);
  }

  /* iterate over mutation table and reorder migration table to match */
  List *new_cellnames = lst_new_ptr(mg->ncells);
  List *new_states = lst_new_int(mg->ncells);
  for (i = 0; i < mm->ncells; i++) {
    String *s = lst_get_ptr(mm->cellnames, i);
    int mig_idx = hsh_get_int(mignamehash, s->chars);
    if (mig_idx == -1)
      die("ERROR: cell '%s' in mutation table not found in migration table.\n",
          s->chars);
    lst_push_ptr(new_cellnames, str_dup(s));
    lst_push_int(new_states, lst_get_int(mg->states, mig_idx));
  }

  /* replace cellnames and states with reordered versions */
  lst_free_strings(mg->cellnames);
  lst_free(mg->cellnames);
  lst_free(mg->states);
  hsh_free(mignamehash);
  mg->cellnames = new_cellnames;
  mg->states = new_states;
}

/* helper function to avoid underflow if some migration rates approach zero */
#define RATE_FLOOR 1.0e-300
static inline double mm_get_floor(MarkovMatrix *M, int i, int j) {
  double p = mm_get(M, i, j);
  return p + RATE_FLOOR; /* note derivative still same as orig */
}

/* compute log likelihood of migration model based on a given tree
   model and migration table for tips.  If branchgrad is non-null, it
   will be populated with the gradient with respect to the individual
   branches of the tree, in post-order */
double mig_compute_log_likelihood(TreeModel *mod, MigTable *mg, 
                                  CrisprMutModel *cprmod, Vector *branchgrad) {

  int i, j, k, nodeidx, cell, state;
  int nstates = mg->nstates; 
  TreeNode *n, *sibling;
  double total_prob = 0;
  List *traversal, *pre_trav;
  double **pL = NULL, **pLbar = NULL;
  double ll = 0;
  double tmp[nstates], root_eqfreqs[nstates];
  Matrix **grad_mat = NULL;
  List **grad_mat_P = NULL;
  MarkovMatrix *par_subst_mat, *sib_subst_mat, *leading_Pt, *lsubst_mat, *rsubst_mat; ;
  Vector *lscale, *lscale_o; /* inside and outside versions */
  Vector *this_deriv_gtr = NULL;

  /* set up "inside" probability matrices for pruning algorithm */
  pL = smalloc(nstates * sizeof(double*));
  for (j = 0; j < nstates; j++)
    pL[j] = smalloc((cprmod->mod->tree->nnodes+1) * sizeof(double));

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = vec_new(mod->tree->nnodes+1); 
  lscale_o = vec_new(mod->tree->nnodes+1); 
  vec_zero(lscale); vec_zero(lscale_o);
  
  if (branchgrad != NULL) {
    if (branchgrad->size != mod->tree->nnodes-1) /* rooted */
      die("ERROR in mig_compute_log_likelihood: size of branchgrad must be 2n-2\n");
    vec_zero(branchgrad);
    /* set up complementary "outside" probability matrices */
    pLbar = smalloc(nstates * sizeof(double*));
    for (j = 0; j < nstates; j++)
      pLbar[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));

    vec_zero(mg->deriv_gtr);
    grad_mat = malloc(mod->tree->nnodes * sizeof(Matrix*));
    for (j = 0; j < mod->tree->nnodes; j++)
      grad_mat[j] = mat_new(nstates, nstates);
    /* each node of the tree needs a list of gradient
       matrices, one for each free GTR parameter */
    grad_mat_P = malloc(mod->tree->nnodes * sizeof(void*));
    for (j = 0; j < mod->tree->nnodes; j++) {
      grad_mat_P[j] = lst_new_ptr(mg->gtr_params->size);
      for (int jj = 0; jj < mg->gtr_params->size; jj++)
        lst_push_ptr(grad_mat_P[j],
                     mat_new(nstates, nstates));
    }
    this_deriv_gtr = vec_new(mg->gtr_params->size);
  }

  mig_update_subst_matrices(mod->tree, mg); /* compute all necessary migration probability
                                       matrices */
  if (cprmod->mod->msa_seq_idx == NULL)
    cpr_build_seq_idx(cprmod->mod, cprmod->mut);

  traversal = tr_postorder(cprmod->mod->tree);
    
  /* this model allows a leading branch to the root of the tree.  We can
     simulate this behavior by setting the root eq freqs equal to
     the conditional distribution at the end of the branch  */
  leading_Pt = lst_get_ptr(mg->Pt, mod->tree->id);
  if (mg->primary_state != -1) { /* force primary state at root */
    for (i = 0; i < nstates; i++)
      root_eqfreqs[i] = mm_get_floor(leading_Pt, mg->primary_state, i);
  }
  else { /* sum over root states */   
    for (i = 0; i < nstates; i++) { /* pseudo root */
      root_eqfreqs[i] = 0;
      for (j = 0; j < nstates; j++) /* actual root */
        root_eqfreqs[i] += vec_get(mg->backgd_freqs, j) *
          mm_get_floor(leading_Pt, j, i);
    }
  }
    
  for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
    n = lst_get_ptr(traversal, nodeidx);

    if (n->lchild == NULL) {
      /* leaf: base case of recursion */
      cell = cprmod->mod->msa_seq_idx[n->id];
      
      if (cell == -1)
        die("ERROR in mig_compute_log_likelihood: leaf '%s' not found in "
            "migration table.\n", n->name);

      state = lst_get_int(mg->states, cell);
      assert(state >= 0 && state < nstates);
      
      for (i = 0; i < nstates; i++) {
        if (i == state)
          pL[i][n->id] = 1;
        else
          pL[i][n->id] = 0;
      }
    }
    else {
      /* general recursive case */
      lsubst_mat = lst_get_ptr(mg->Pt, n->lchild->id);
      rsubst_mat = lst_get_ptr(mg->Pt, n->rchild->id);

      /* do this in a way that handles scaling.  first compute
         unnormalized inside values */
      for (i = 0; i < nstates; i++) {
        double totl = 0.0, totr = 0.0;

        for (j = 0; j < nstates; j++)
          totl += pL[j][n->lchild->id] *
                  mm_get(lsubst_mat, i, j);

        for (k = 0; k < nstates; k++)
          totr += pL[k][n->rchild->id] *
                  mm_get(rsubst_mat, i, k);

        pL[i][n->id] = totl * totr;
      }

      /* nodewise max-normalization across states */
      double maxv = 0.0;
      for (i = 0; i < nstates; i++)
        if (pL[i][n->id] > maxv)
          maxv = pL[i][n->id];

      /* propagate scaling from children */
      vec_set(lscale, n->id,
              vec_get(lscale, n->lchild->id) +
              vec_get(lscale, n->rchild->id));

      if (maxv > 0.0) {
        /* normalize and update scale */
        for (i = 0; i < nstates; i++)
          pL[i][n->id] /= maxv;

        vec_set(lscale, n->id,
                vec_get(lscale, n->id) + log(maxv));
      }
      else 

      /* zero out tiny values to save time later */
      for (i = 0; i < nstates; i++)
        if (pL[i][n->id] < REL_CUTOFF)
          pL[i][n->id] = 0.0;
    }
  }
  
  /* termination */
  total_prob = 0;
  for (i = 0; i < nstates; i++)
    total_prob += pL[i][mod->tree->id] * root_eqfreqs[i];
  if (total_prob <= 0.0)
    total_prob = REL_CUTOFF;
  
  ll += (log(total_prob) + vec_get(lscale, mod->tree->id));

  /* to compute gradients efficiently, need to make a second pass
     across the tree to compute "outside" probabilities */
  if (branchgrad != NULL) {
    double expon = 0;
    pre_trav = tr_preorder(mod->tree);

    for (nodeidx = 0; nodeidx < lst_size(pre_trav); nodeidx++) {
      n = lst_get_ptr(pre_trav, nodeidx);

      if (n->parent == NULL) { /* base case */
        double maxv = 0.0;
        for (i = 0; i < nstates; i++) {
          pLbar[i][n->id] = root_eqfreqs[i];
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
        sibling = (n == n->parent->lchild ? n->parent->rchild : n->parent->lchild);
        par_subst_mat = lst_get_ptr(mg->Pt, n->id);
        sib_subst_mat = lst_get_ptr(mg->Pt, sibling->id);

        /* first form tmp[j] = sum_k pLbar(parent=j) * pL(sibling=k) * P_sib(j,k) */
        for (j = 0; j < nstates; j++) {
          tmp[j] = 0.0;
          double a = pLbar[j][n->parent->id];

          if (a == 0.0) continue;

          for (k = 0; k < nstates; k++) {
            double b = pL[k][sibling->id];
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
                vec_get(lscale, sibling->id));

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
    for (nodeidx = 0; nodeidx < lst_size(mod->tree->nodes); nodeidx++) {
      TreeNode *par, *sibling;
      double base_prob = total_prob, deriv;
        
      n = lst_get_ptr(mod->tree->nodes, nodeidx);
      par = n->parent;
        
      if (par == NULL) 
        continue;
       
      sibling = (n == n->parent->lchild ?
                 n->parent->rchild : n->parent->lchild);

      sib_subst_mat = lst_get_ptr(mg->Pt, sibling->id);

      /* this part is just a constant to propagate through to the
         derivative */
      for (i = 0; i < nstates; i++) {  /* parent */
        tmp[i] = 0;
        for (k = 0; k < nstates; k++)  /* sibling */
          tmp[i] += pL[k][sibling->id] * mm_get_floor(sib_subst_mat, i, k);
      }

      if (n != mod->tree->rchild) { /* skip branch to right of root because unrooted */
        /* calculate derivative analytically */
        deriv = 0;
        mig_grad_REV_dt(mg, grad_mat[n->id], n->dparent); /* FIXME: customize */
          
        for (i = 0; i < nstates; i++)   
          for (j = 0; j < nstates; j++)    
            deriv +=  tmp[i] * pLbar[i][par->id] * pL[j][n->id] * mat_get(grad_mat[n->id], i, j);

        /* adjust for all relevant scale terms; do everything in log space */
        expon = -vec_get(lscale, mod->tree->id)
          + vec_get(lscale, sibling->id) + vec_get(lscale_o, par->id)
          + vec_get(lscale, n->id) - log(base_prob);
        /* note division by base_prob because we need deriv of log P */

        /* avoid overflow */
        if (expon > 700.0) expon = 700.0;
        if (expon < -745.0) expon = -745.0;
          
        deriv *= exp(expon);
        assert(isfinite(deriv));
                  
        vec_set(branchgrad, n->id, vec_get(branchgrad, n->id) + deriv);
      }

      /* we need partial derivatives for migration rates also;
         they have to be aggregated across all branches */
      mig_grad_REV_dr(mg, grad_mat_P[n->id], n->dparent);  /* FIXME: customize */
      /* loop over rate parameters */
      for (int pidx = 0; pidx < mg->gtr_params->size; pidx++) {
        double pderiv = 0; /* partial deriv wrt this param */
        Matrix *dP_dr = lst_get_ptr(grad_mat_P[n->id], pidx);
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
      vec_plus_eq(mg->deriv_gtr, this_deriv_gtr);
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
    for (j = 0; j < mod->tree->nnodes; j++) {
      List *gmats = grad_mat_P[j];
      for (int jj = 0; jj < lst_size(gmats); jj++)
        mat_free(lst_get_ptr(gmats, jj));
      lst_free(gmats);
    }
    free(grad_mat_P);
    vec_free(this_deriv_gtr);
  }

  vec_free(lscale);
  vec_free(lscale_o);
  
  return ll;
}

/* set P = exp(Qt) for each branch in the tree model */
void mig_update_subst_matrices(TreeNode *tree, MigTable *mg) {
  static int last_nnodes = -1;

  if (mg->Pt != NULL && tree->nnodes != last_nnodes) { /* this can happen with deduplication */    
    for (int nodeidx = 0; nodeidx < lst_size(mg->Pt); nodeidx++)
      mm_free(lst_get_ptr(mg->Pt, nodeidx));
    mg->Pt = NULL; /* force reallocation */
  }

  if (mg->Pt == NULL) { /* first time through or realloc */
    mg->Pt = lst_new_ptr(tree->nnodes);
    for (int nodeidx = 0; nodeidx < tree->nnodes; nodeidx++) {
      MarkovMatrix *P = mm_new(mg->nstates, NULL, DISCRETE);
      mm_set_eigentype(P, REAL_NUM);
      lst_push_ptr(mg->Pt, P);
    }
    last_nnodes = tree->nnodes;
  }
  
  for (int nodeidx = 0; nodeidx < tree->nnodes; nodeidx++) {
    TreeNode *n = lst_get_ptr(tree->nodes, nodeidx);
    MarkovMatrix *P = lst_get_ptr(mg->Pt, nodeidx);
    mm_exp(P, mg->rate_matrix, n->dparent);

    /* ensure no negative values due to numerical error */
    for (int i = 0; i < mg->nstates; i++) {
      for (int j = 0; j < mg->nstates; j++) {
        double p = mm_get(P, i, j);
        if (p < 0.0)
          mm_set(P, i, j, 0.0);
      }
    }
  }
}

/* helper function to sample states at internal nodes.  Given an array of
   doubles of dimension nstates, sample in proportion to those values and return
   an index between 0 and nstates-1.  Original probabilities do not need to be
   normalized */
static inline
int mig_sample_state(double *probs, int nstates) {
  double total = 0, cumprob = 0;
  for (int i = 0; i < nstates; i++)
    total += probs[i];
  assert(total > 0.0);
  double r = unif_rand() * total;
  for (int i = 0; i < nstates; i++) {
    cumprob += probs[i];
    if (r <= cumprob)
      return i;
  }
  return nstates - 1; /* should not get here */
}

/* given a tree and migration table, sample states at internal
   nodes of the tree.  Populates a list parallel to tree->nodes
   whose elements are state indices */
void mig_sample_states(TreeNode *tree, MigTable *mg, 
                       CrisprMutModel *cprmod, List *state_samples) {

  int i, j, k, nodeidx, cell, state;
  int nstates = mg->nstates; 
  TreeNode *n;
  List *traversal, *pre_trav;
  double **pL = NULL;
  double scaling_threshold = sqrt(DBL_MIN) * 1.0e10;  /* need some padding */
  double lscaling_threshold = log(scaling_threshold);
  MarkovMatrix *par_subst_mat, *leading_Pt, *lsubst_mat, *rsubst_mat; ;
  Vector *lscale; /* inside and outside versions */
  unsigned int rescale;

  /* initialize state_samples */
  lst_clear(state_samples);
  for (i = 0; i < tree->nnodes + 1; i++)
    lst_push_int(state_samples, -1);
  
  /* set up "inside" probability matrices */
  pL = smalloc(nstates * sizeof(double*));
  for (j = 0; j < nstates; j++)
    pL[j] = smalloc((tree->nnodes+1) * sizeof(double));

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = vec_new(tree->nnodes+1); 
  vec_zero(lscale);
  
  mig_update_subst_matrices(tree, mg); /* compute all necessary migration probability
                                          matrices */
  if (cprmod->mod->msa_seq_idx == NULL)
    cpr_build_seq_idx(cprmod->mod, cprmod->mut);

  traversal = tr_postorder(tree);

  leading_Pt = lst_get_ptr(mg->Pt, tree->id);

  for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
    n = lst_get_ptr(traversal, nodeidx);

    if (n->lchild == NULL) {
      /* leaf: base case of recursion */
      cell = cprmod->mod->msa_seq_idx[n->id];
      
      if (cell == -1)
        die("ERROR in mig_sample_states: leaf '%s' not found in "
            "migration table.\n", n->name);

      state = lst_get_int(mg->states, cell);
      assert(state >= 0 && state < nstates);
      
      for (i = 0; i < nstates; i++) {
        if (i == state)
          pL[i][n->id] = 1;
        else
          pL[i][n->id] = 0;
      }

      /* we can set the state samples for leaves now */
      lst_set_int(state_samples, n->id, state);
    }
    else {
      /* general recursive case */
      lsubst_mat = lst_get_ptr(mg->Pt, n->lchild->id);
      rsubst_mat = lst_get_ptr(mg->Pt, n->rchild->id);

      rescale = FALSE;
      for (int pass = 0; pass < 2 && (pass == 0 || rescale); pass++) {
        for (i = 0; i < nstates; i++) {
          double totl = 0.0, totr = 0.0;
        for (j = 0; j < nstates; j++)
          totl += pL[j][n->lchild->id] *
            mm_get_floor(lsubst_mat, i, j);
          
        for (k = 0; k < nstates; k++)
          totr += pL[k][n->rchild->id] *
            mm_get_floor(rsubst_mat, i, k);

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
    }
  }

  /* Now pass from root to leaves and sample notes based on smpled parent and
     inside probabilities */
  pre_trav = tr_preorder(tree);
  double *sampdens = smalloc(nstates * sizeof(double));
  for (nodeidx = 0; nodeidx < lst_size(pre_trav); nodeidx++) {
    n = lst_get_ptr(pre_trav, nodeidx);

    if (n->parent == NULL) { /* base case */
        int sampled_primary_state;
        int origin_idx = tree->nnodes;

        if (mg->primary_state != -1) {
            /* Use the forced primary state at the origin node */
            sampled_primary_state = mg->primary_state;  
        }
        else {
            double *startdens = smalloc(nstates * sizeof(double));

            /* Sample the origin state from its posterior given the subtree data */
            for (j = 0; j < nstates; j++) {
                startdens[j] = 0.0;
                for (i = 0; i < nstates; i++) {
                    startdens[j] += vec_get(mg->backgd_freqs, j) *
                                    mm_get_floor(leading_Pt, j, i) *
                                    pL[i][n->id];
                }
            }

            sampled_primary_state = mig_sample_state(startdens, nstates);
            sfree(startdens);
        }

        lst_set_int(state_samples, origin_idx, sampled_primary_state);

        /* Sample the root state from its posterior given the origin state
          and the subtree data (via inside likelihood pL) */
        for (i = 0; i < nstates; i++)
            sampdens[i] = mm_get_floor(leading_Pt, sampled_primary_state, i) *
                          pL[i][n->id];

        state = mig_sample_state(sampdens, nstates);
        lst_set_int(state_samples, n->id, state);
    }
    else if (n->lchild == NULL)  /* leaf: already handled */
      continue;
    else { /* recursive case */
      int parstate = lst_get_int(state_samples, n->parent->id);
      par_subst_mat = lst_get_ptr(mg->Pt, n->id);

      for (i = 0; i < nstates; i++)
        sampdens[i] = pL[i][n->id] * mm_get_floor(par_subst_mat, parstate, i);;
      state = mig_sample_state(sampdens, nstates);

      lst_set_int(state_samples, n->id, state);
    }
  }
  
  for (j = 0; j < nstates; j++)
    sfree(pL[j]);
  sfree(pL);
  
  vec_free(lscale);
  sfree(sampdens);
}

/* based on a tree and list of states at all nodes, obtain a
   multigraph representing migration events and their times.
   
   TODO: Record the origin branch height in the times and the
   state transition if the origin state != root state.
   */
struct mdag *mig_get_graph(TreeNode *tree, MigTable *mg, List *state_samples) {
  MultiDAG *g = mdag_new(mg);

  /* first compute height of each node */
  List *traversal = tr_preorder(tree);
  Vector *heights = vec_new(tree->nnodes);
  vec_set_all(heights, 0.0);
  for (int nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
    TreeNode *n = lst_get_ptr(traversal, nodeidx);
    if (n->parent == NULL) /* root */
      vec_set(heights, n->id, 0.0);
    else { 
      double par_h = vec_get(heights, n->parent->id);
      vec_set(heights, n->id, par_h + n->dparent);
    }
  }  
 
  for (int nodeidx = 0; nodeidx < tree->nnodes; nodeidx++) {
    TreeNode *n = lst_get_ptr(tree->nodes, nodeidx);
    if (n->parent == NULL)
      continue;
    int childstate = lst_get_int(state_samples, n->id);
    int parstate = lst_get_int(state_samples, n->parent->id);
    if (childstate != parstate) {
      double start_time = vec_get(heights, n->parent->id);
      double end_time = vec_get(heights, n->id);
      mdag_add_edge(g, parstate, childstate, start_time, end_time); 
    }
  }
  return g;
}

struct mdag *mig_sample_graph(TreeNode *tree, MigTable *mg, 
                             CrisprMutModel *cprmod) {
  List *state_samples = lst_new_int(tree->nnodes);
  mig_sample_states(tree, mg, cprmod, state_samples);
  struct mdag *g = mig_get_graph(tree, mg, state_samples);
  lst_free(state_samples);
  return g;
}

/* Print a NEXUS file in which each node is labeled with its sampled
   state using BEAST-style notation embedded in the node labels */
void mig_print_labeled_nexus(TreeNode *tree, FILE *outf, MigTable *mg,
                             List *state_samples) {
  int nn = tree->nnodes;

  /* 1) Gather TAXA (leaf names) in any traversal; leaves are guaranteed named/unique */
  List *trav = tree->nodes;
  List *taxa = lst_new_ptr(64);
  for (int i = 0; i < nn; i++) {
    TreeNode *n = lst_get_ptr(trav, i);
    if (n->lchild == NULL && n->rchild == NULL)
      lst_push_ptr(taxa, n->name);
  }

  /* 2) Temporarily append annotations to node names: "<name>[&state=STATE]" */
  String **saved_names = (String**)smalloc(nn * sizeof(String*));
  for (int i = 0; i < nn; i++)
    saved_names[i] = NULL;
 
  for (int i = 0; i < lst_size(trav); i++) {
    TreeNode *n = lst_get_ptr(trav, i);
    int state = lst_get_int(state_samples, n->id);
    assert(state >= 0 && state < lst_size(mg->statenames));

    /* save original name once per node so we can restore later */
    if (saved_names[n->id] == NULL)
      saved_names[n->id] = str_new_charstr(n->name);

    String *st = str_dup((String*)lst_get_ptr(mg->statenames, state));

    const char *orig = n->name;
    String *tmpS = str_new((int)strlen(orig) + 11 /*"[&state="*/ + st->length + 1);
    if (*orig) str_append_charstr(tmpS, orig);
    str_append_charstr(tmpS, "[&state=");
    str_append(tmpS, st);
    str_append_char(tmpS, ']');

    strncpy(n->name, tmpS->chars, sizeof(n->name) - 1);
    n->name[sizeof(n->name) - 1] = '\0';

    str_free(st);
    str_free(tmpS);
  }

  /* 3) Print NEXUS file with modified names */
  const char *tname = "TREE1";

  fprintf(outf, "#NEXUS\n\n");

  /* TAXA block with original leaf names (no annotations) */
  fprintf(outf, "BEGIN TAXA;\n");
  fprintf(outf, "  DIMENSIONS NTAX=%d;\n", lst_size(taxa));
  fprintf(outf, "  TAXLABELS\n");
  for (int i = 0; i < lst_size(taxa); i++) {
    const char *lab = (const char*)lst_get_ptr(taxa, i);
    fprintf(outf, "    %s\n", lab ? lab : "taxon");
  }
  fprintf(outf, "  ;\nEND;\n\n");

  /* TREES block: prefix line, then let tr_print write the Newick + ';' + '\n' */
  fprintf(outf, "BEGIN TREES;\n");
  fprintf(outf, "  TREE %s = [&R] ", tname);
  tr_print(outf, tree, /*show_branch_lengths=*/1);
  fprintf(outf, "END;\n\n");

  /* 4) Restore original names and free temporaries */
  for (int i = 0; i < lst_size(trav); i++) {
  TreeNode *n = lst_get_ptr(trav, i);
  if (saved_names[n->id]) {
    strncpy(n->name, saved_names[n->id]->chars, sizeof(n->name) - 1);
    n->name[sizeof(n->name) - 1] = '\0';
    str_free(saved_names[n->id]);
    saved_names[n->id] = NULL;
  }
}
  sfree(saved_names);
  lst_free(taxa);
}

/* Print a NEXUS file with a single header and multiple TREE lines,
   where each node in each tree is labeled with its sampled state using
   BEAST-style notation embedded in the node labels. 
   
   TODO: Print the origin node state, but unsure how this works with 
   standard NEXUS format */
void mig_print_set_labeled_nexus(List *tree_lst, FILE *outf, MigTable *mg,
                                 List *statesamps_lst) {
  assert(lst_size(tree_lst) == lst_size(statesamps_lst));

  /* Use the first model to define TAXA (leaf names) */
  TreeNode *tree0 = (TreeNode*)lst_get_ptr(tree_lst, 0);
  int nn0 = tree0->nnodes;

  /* Gather TAXA from the first tree */
  List *trav0 = tree0->nodes;
  List *taxa = lst_new_ptr(64);
  for (int i = 0; i < nn0; i++) {
    TreeNode *n = lst_get_ptr(trav0, i);
    if (n->lchild == NULL && n->rchild == NULL)
      lst_push_ptr(taxa, n->name);
  }

  /* Header and TAXA block (printed once) */
  fprintf(outf, "#NEXUS\n\n");
  fprintf(outf, "BEGIN TAXA;\n");
  fprintf(outf, "  DIMENSIONS NTAX=%d;\n", lst_size(taxa));
  fprintf(outf, "  TAXLABELS\n");
  for (int i = 0; i < lst_size(taxa); i++) {
    const char *lab = (const char*)lst_get_ptr(taxa, i);
    fprintf(outf, "    %s\n", lab ? lab : "taxon");
  }
  fprintf(outf, "  ;\nEND;\n\n");

  /* TREES block with one TREE per sample */
  fprintf(outf, "BEGIN TREES;\n");

  int nsamp = lst_size(tree_lst);
  for (int s = 0; s < nsamp; s++) {
    TreeNode *tree = (TreeNode*)lst_get_ptr(tree_lst, s);
    List *state_samples = (List*)lst_get_ptr(statesamps_lst, s);
    int nn = tree->nnodes;

    /* Traverse nodes for this tree */
    List *trav = tree->nodes;

    /* Temporarily append annotations to node names: "<name>[&state=STATE]" */
    String **saved_names = (String**)smalloc(nn * sizeof(String*));
    for (int i = 0; i < nn; i++) saved_names[i] = NULL;

    for (int i = 0; i < lst_size(trav); i++) {
      TreeNode *n = lst_get_ptr(trav, i);
      int state = lst_get_int(state_samples, n->id);
      assert(state >= 0 && state < lst_size(mg->statenames));

      if (saved_names[n->id] == NULL)
        saved_names[n->id] = str_new_charstr(n->name);

      String *st = str_dup((String*)lst_get_ptr(mg->statenames, state));

      const char *orig = n->name;
      String *tmpS = str_new((int)strlen(orig) + 11 /*"[&state="*/ + st->length + 1);
      if (*orig) str_append_charstr(tmpS, orig);
      str_append_charstr(tmpS, "[&state=");
      str_append(tmpS, st);
      str_append_char(tmpS, ']');

      strncpy(n->name, tmpS->chars, sizeof(n->name) - 1);
      n->name[sizeof(n->name) - 1] = '\0';

      str_free(st);
      str_free(tmpS);
    }

    /* Print one TREE line for this sample */
    fprintf(outf, "  TREE sample_%d = [&R] ", s + 1);
    tr_print(outf, tree, /*show_branch_lengths=*/1);

    /* Restore original names for this tree */
    for (int i = 0; i < lst_size(trav); i++) {
      TreeNode *n = lst_get_ptr(trav, i);
      if (saved_names[n->id]) {
        strncpy(n->name, saved_names[n->id]->chars, sizeof(n->name) - 1);
        n->name[sizeof(n->name) - 1] = '\0';
        str_free(saved_names[n->id]);
        saved_names[n->id] = NULL;
      }
    }
    sfree(saved_names);
  }

  fprintf(outf, "END;\n\n");

  lst_free(taxa);
}

/* print dot file based on list of trees and list of state labels */
void mig_print_set_dot(List *tree_lst, FILE *outf, MigTable *mg,
                       List *statesamps_lst) {
  assert(lst_size(tree_lst) == lst_size(statesamps_lst));
  MultiDAGSet *set = mdag_set_new();
  for (int i = 0; i < lst_size(tree_lst); i++) {
    TreeNode *tree = lst_get_ptr(tree_lst, i);
    List *state_samples = (List*)lst_get_ptr(statesamps_lst, i);
    struct mdag *g = mig_get_graph(tree, mg, state_samples);
    mdag_add_to_set(set, g);
  }
  mdag_set_print_dot(set, outf);
  mdag_set_free(set);
}

/* print CSV file with edgewise migration-event probabilities across a set of trees */
void mig_print_set_edgewise_csv(List *tree_lst, FILE *outf, MigTable *mg,
                                List *statesamps_lst) {
  int i, j;
  int ntrees;
  List *event_names, *event_counts;

  assert(lst_size(tree_lst) == lst_size(statesamps_lst));
  ntrees = lst_size(tree_lst);

  event_names = lst_new_ptr(50);
  event_counts = lst_new_int(50);

  /* Iterate over all trees individually */
  for (i = 0; i < ntrees; i++) {
    TreeNode *tree = lst_get_ptr(tree_lst, i);  /* Get the root of the i-th tree */
    List *state_samples = (List*)lst_get_ptr(statesamps_lst, i);  /* Get the list of states (tissues) for the i-th tree */
    int origin_idx = tree->nnodes;
    int *pair_event_idx = smalloc(mg->nstates * mg->nstates * sizeof(int)); /* Initialize array to track event indices */

    /* Initialize event indices to zero */
    for (j = 0; j < mg->nstates * mg->nstates; j++)
      pair_event_idx[j] = 0;

    /* Iterate over all nodes in the tree */
    for (j = 0; j < tree->nnodes; j++) {
      TreeNode *n = lst_get_ptr(tree->nodes, j);
      int parstate;
      int childstate;

      /* Get the parent state (tissue) for the branch */
      if (n->parent == NULL) {
        /* Get the sampled state for the origin node on the leading branch */
        parstate = lst_get_int(state_samples, origin_idx);
      } 
      else {
        /* Get the sampled state for all other branches */
        parstate = lst_get_int(state_samples, n->parent->id);
      }

      /* Get the sampled child state (tissue) for the branch */
      childstate = lst_get_int(state_samples, n->id);

      /* Skip self migrations (same tissue) */
      if (childstate == parstate)
        continue;

      /* Increment the event index for this pair of states (tissues) */
      int pair_idx = parstate * mg->nstates + childstate;
      int event_idx = ++pair_event_idx[pair_idx];

      /* Generate migration event name with the event index */
      String *parname = (String*)lst_get_ptr(mg->statenames, parstate);
      String *childname = (String*)lst_get_ptr(mg->statenames, childstate);
      String *event_name = str_new(STR_MED_LEN);
      str_append(event_name, parname);
      str_append_charstr(event_name, "_");
      str_append(event_name, childname);
      str_append_charstr(event_name, "_");
      str_append_int(event_name, event_idx);

      /* Check if this event name already exists in the consensus */
      int name_pos = -1;
      for (int k = 0; k < lst_size(event_names); k++) {
        String *cand = (String*)lst_get_ptr(event_names, k);
        if (!strcmp(cand->chars, event_name->chars)) {
          name_pos = k;
          break;
        }
      }

      if (name_pos == -1) {
        /* Add the event name to the consensus and initialize the count to 1 */
        lst_push_ptr(event_names, event_name);
        lst_push_int(event_counts, 1);
      }
      else {
        /* Increment the count for an existing event in the consensus */
        int c = lst_get_int(event_counts, name_pos);
        lst_set_int(event_counts, name_pos, c + 1);
        str_free(event_name);
      }
    }

    sfree(pair_event_idx);
  }

  /* Print the header */
  fprintf(outf, "migration_event,edgewise_probability\n");

  /* Print events from highest to lowest probability (equivalently, count) */
  int nevents = lst_size(event_names);
  int *printed = smalloc(nevents * sizeof(int));
  for (i = 0; i < nevents; i++)
    printed[i] = 0;

  for (i = 0; i < nevents; i++) {
    int best_idx = -1;
    int best_count = -1;
    for (j = 0; j < nevents; j++) {
      if (printed[j])
        continue;
      int c = lst_get_int(event_counts, j);
      if (c > best_count) {
        best_count = c;
        best_idx = j;
      }
    }

    if (best_idx == -1)
      break;

    printed[best_idx] = 1;
    String *event_name = (String*)lst_get_ptr(event_names, best_idx);
    double prob = (ntrees > 0 ? (double)best_count / ntrees : 0.0);
    fprintf(outf, "%s,%.10g\n", event_name->chars, prob);
  }
  sfree(printed);

  lst_free_strings(event_names);
  lst_free(event_names);
  lst_free(event_counts);
}

/***************************************************************************
 Functions adapted from phast_subst_mods.c to handle migration models 
 ***************************************************************************/
void mig_set_REV_matrix(MigTable *mg, Vector *params) {
  int i, j, start_idx = 0;
  if (mg->backgd_freqs == NULL)
    die("mig_set_REV_matrix: mg->backgd_freqs is NULL\n");
  for (i = 0; i < mg->rate_matrix->size; i++) {
    double rowsum = 0;
    for (j = i+1; j < mg->rate_matrix->size; j++) {
      double val;
      val = vec_get(params, start_idx);
      mm_set(mg->rate_matrix, i, j,
             val * vec_get(mg->backgd_freqs, j));
      mm_set(mg->rate_matrix, j, i,
             val * vec_get(mg->backgd_freqs, i));
      rowsum += (val * vec_get(mg->backgd_freqs, j));
      lst_clear(mg->rate_matrix_param_row[start_idx]);
      lst_clear(mg->rate_matrix_param_col[start_idx]);
      lst_push_int(mg->rate_matrix_param_row[start_idx], i);
      lst_push_int(mg->rate_matrix_param_col[start_idx], j);
      lst_push_int(mg->rate_matrix_param_row[start_idx], j);
      lst_push_int(mg->rate_matrix_param_col[start_idx], i);

      start_idx++;
    }
    for (j = 0; j < i; j++)
      rowsum += mm_get(mg->rate_matrix, i, j);
    mm_set(mg->rate_matrix, i, i, -1 * rowsum);
  }
  /* NOTE: do not scale in this case; have to allow migration rate to
     be decoupled from mutation rate */
  mm_diagonalize(mg->rate_matrix);
}

void mig_scale_rate_matrix(MigTable *mg) {
  double scale = 0;
  for (int i = 0; i < mg->rate_matrix->size; i++) {
    double rowsum = 0;
    for (int j = 0; j < mg->rate_matrix->size; j++) 
      if (j != i) rowsum += mm_get(mg->rate_matrix, i, j);
    scale += (vec_get(mg->backgd_freqs, i) * rowsum);
  }
  mm_scale(mg->rate_matrix, 1.0/scale);
}

/* dP/dt for REV */
void mig_grad_REV_dt(MigTable *mg, Matrix *grad, double t) {
  double g_ij, s_ik, sprime_kj, lambda_k;
  MarkovMatrix *Q = mg->rate_matrix;

  if (t == 0) {
    mat_copy(grad, Q->matrix); /* at t=0, dP/dt = Q */    
    return;
  }
  
  if (Q->evec_matrix_r == NULL || Q->evals_r == NULL ||
      Q->evec_matrix_inv_r == NULL) {
    mm_diagonalize(Q);
    if (Q->diagonalize_error) {
      mat_print(Q->matrix, stderr);
      die("ERROR in mig_grad_REV_dt: rate matrix could not be diagonalized.\n");
    }
  }
  for (int i = 0; i < Q->size; i++) {
    for (int j = 0; j < Q->size; j++) {
      g_ij = 0;
      for (int k = 0; k < Q->size; k++) {
        s_ik = mat_get(Q->evec_matrix_r, i, k);
        sprime_kj = mat_get(Q->evec_matrix_inv_r, k, j);
        lambda_k = vec_get(Q->evals_r, k);
        g_ij += s_ik * lambda_k * exp(lambda_k * t) * sprime_kj;
        /* see Siepel & Hausser 2004 Eq C.10 but in this case we omit
           the denominator and avoid summing over elements (handled by
           calling code) */
      }
      mat_set(grad, i, j, g_ij);
    }
  }  
}

/* dP/dr for REV, where r is a free parameter corresponding in the
   rate matrix.  Fills dQ_dr, a list of matrices, the ith element of
   which corresponds to the ith free parameter */
void mig_grad_REV_dr(MigTable *mg, List *dP_dr_lst, double t) {

  /* this code is adapted from compute_grad_em_exact in
     phast_fit_em.c */
  int i, j, k, l, m, idx, lidx, orig_size;
  int nstates = mg->nstates;
  List *erows = lst_new_int(4), *ecols = lst_new_int(4), 
    *distinct_rows = lst_new_int(2);

  double **tmpmat = smalloc(nstates * sizeof(double *));
  double **sinv_dq_s = smalloc(nstates * sizeof(double *));
  double **dq = smalloc(nstates * sizeof(double *));
  double **f = smalloc(nstates * sizeof(double *));
  for (i = 0; i < nstates; i++) {
    dq[i] = smalloc(nstates * sizeof(double));
    tmpmat[i] = smalloc(nstates * sizeof(double));
    sinv_dq_s[i] = smalloc(nstates * sizeof(double));
    f[i] = smalloc(nstates * sizeof(double));
  }

  MarkovMatrix *Q = mg->rate_matrix;

  if (Q->evec_matrix_r == NULL || Q->evals_r == NULL ||
      Q->evec_matrix_inv_r == NULL) {
    mm_diagonalize(Q);
    if (Q->diagonalize_error) {
      mat_print(Q->matrix, stderr);
      die("ERROR in mig_grad_REV_dr: rate matrix could not be diagonalized.\n");
    }
  }
  
  for (idx = 0; idx < lst_size(dP_dr_lst); idx++) {

    Matrix *dP_dr = (Matrix *)lst_get_ptr(dP_dr_lst, idx);
    mat_zero(dP_dr);
    
    for (i = 0; i < nstates; i++) 
      for (j = 0; j < nstates; j++) 
        dq[i][j] = tmpmat[i][j] = sinv_dq_s[i][j] = 0;

    /* element coords (rows/col pairs) at which current param appears in Q */
    lst_cpy(erows, mg->rate_matrix_param_row[idx]);
    lst_cpy(ecols, mg->rate_matrix_param_col[idx]);
    if (lst_size(erows) != lst_size(ecols))
      die("ERROR mig_grad_REV_dr: size of erows (%i) does not match size of "
          "ecols (%i)\n",
          lst_size(erows), lst_size(ecols));

    /* set up dQ, the partial deriv of Q wrt the current param */
    lst_clear(distinct_rows);
    for (i = 0, orig_size = lst_size(erows); i < orig_size; i++) {
      l = lst_get_int(erows, i); 
      m = lst_get_int(ecols, i);

      if (dq[l][m] != 0)    /* row/col pairs should be unique */
        die("ERROR tm_grad_REV_dr: dq[%i][%i] should be zero but is %f\n",
            l, m, dq[l][m]);

      dq[l][m] = vec_get(mg->backgd_freqs, m);      
      if (dq[l][m] == 0) continue; 

      /* keep track of distinct rows and cols with non-zero entries */
      /* also add diagonal elements to 'rows' and 'cols' lists, as
         necessary */
      if (dq[l][l] == 0) {      /* new row */
        lst_push_int(distinct_rows, l);
        lst_push_int(erows, l);
        lst_push_int(ecols, l);
      }

      dq[l][l] -= dq[l][m]; /* row sums to zero */
    }

    /* compute S^-1 dQ S */
    for (lidx = 0; lidx < lst_size(erows); lidx++) {
      i = lst_get_int(erows, lidx);
      k = lst_get_int(ecols, lidx);
      for (j = 0; j < nstates; j++)
        tmpmat[i][j] += mat_get(Q->evec_matrix_r, k, j) * dq[i][k];
    }

    for (lidx = 0; lidx < lst_size(distinct_rows); lidx++) {
      k = lst_get_int(distinct_rows, lidx);
      for (i = 0; i < nstates; i++) {
        for (j = 0; j < nstates; j++) {
          sinv_dq_s[i][j] += mat_get(Q->evec_matrix_inv_r, i, k) * tmpmat[k][j];
        }
      }
    }

    /* set up Schadt and Lange's F matrix */
    for (i = 0; i < nstates; i++) {
      for (j = 0; j < nstates; j++) {
        if (fabs(vec_get(Q->evals_r, i) - vec_get(Q->evals_r, j)) < 1e-8)
          f[i][j] = exp((vec_get(Q->evals_r, i)) * t) * t;
        else
          f[i][j] = (exp((vec_get(Q->evals_r, i)) * t) 
                     - exp((vec_get(Q->evals_r, j)) * t)) /
            ((vec_get(Q->evals_r, i)) - (vec_get(Q->evals_r, j)));
      }
    }

    /* compute (F o S^-1 dQ S) S^-1 */
    for (i = 0; i < nstates; i++) {
      for (j = 0; j < nstates; j++) {
        tmpmat[i][j] = 0;
        for (k = 0; k < nstates; k++) 
          tmpmat[i][j] += f[i][k] * sinv_dq_s[i][k] *
            mat_get(Q->evec_matrix_inv_r, k, j);
      }
    }

    /* compute S (F o S^-1 dQ S) S^-1 */
    for (i = 0; i < nstates; i++) {
      for (j = 0; j < nstates; j++) {
        double partial_p = 0;        
        for (k = 0; k < nstates; k++) 
          partial_p += mat_get(Q->evec_matrix_r, i, k) * tmpmat[k][j];
        mat_set(dP_dr, i, j, partial_p);
      }
    }
  }

  /* free allocated memory */
  for (i = 0; i < nstates; i++) {
    sfree(dq[i]); sfree(tmpmat[i]); sfree(sinv_dq_s[i]); sfree(f[i]);
  }
  sfree(dq); sfree(tmpmat); sfree(sinv_dq_s); sfree(f);
  lst_free(erows); lst_free(ecols); lst_free(distinct_rows);
}

