/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* recursive backprop through NJ algorithm */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>

/* map the indices of two taxa, i, j to a unique index
   for their pairwise distance.  Unique index will fall densely
   between 0 and n-choose-2 - 1 */
int nj_i_j_to_dist(int i, int j, int n) {
  double ii, jj;
  if (i < j) { ii = i; jj = j; }
  else { ii = j; jj = i; }
  return ((2*n - ii - 1)*ii / 2 + (jj - ii - 1));
}

/* reverse the mapping above: map an index for a pairwise distance to
   the indices for two sequences (s.t. j > i) */
void nj_dist_to_i_j(int pwidx, int *i, int *j, int n) {
  int rowstart;
  *i = 0;
  while (pwidx >= (2*n - (*i) - 1) * (*i) / 2 + (n - (*i) - 1))
    (*i)++;
  rowstart = (2*n - (*i) - 1) * (*i) / 2;
  *j = *i + 1 + (pwidx - rowstart);
}

/* function used inside NJ algorithm to enable backpropagation of
   derivatives through the algorithm.  Jk is a matrix such that
   element [ij][ab] represents the partial derivative of the distance
   between i and j on iteration k to the distance between a and b at
   the start of the algorithm. The next value, Jnext, is defined
   recursively from Jk.  The last set of identified neighbors are
   denoted f and g, and u denotes the new node created to replace f
   and g. */
void nj_backprop(double *Jk, double *Jnext, int n, int f, int g, int u,
                 Vector *active) {
  int i, a, b;
  int total_nodes = 2*n - 2; /* total possible in final tree */
  int npairs_large = (total_nodes * (total_nodes - 1)) / 2;
  int npairs_small = (n * (n - 1)) / 2;
  
  /* most of Jk will be unchanged so start by copying the whole thing
     efficiently */
  memcpy(Jnext, Jk, npairs_large * npairs_small * sizeof(double));
  
  /* now update distances involving new node u */
  for (i = 0; i < total_nodes; i++) {
    if (vec_get(active, i) == FALSE || i == f || i == g || i == u) continue;  

    int idx_ui = nj_i_j_to_dist(u, i, total_nodes);
    int idx_fi = nj_i_j_to_dist(f, i, total_nodes);
    int idx_gi = nj_i_j_to_dist(g, i, total_nodes);
    int idx_fg = nj_i_j_to_dist(f, g, total_nodes);

    for (a = 0; a < n; a++) {
      for (b = a + 1; b < n; b++) {
        int idx_ab = nj_i_j_to_dist(a, b, n);

        /* recursive update rule for new distance from u to i */
        Jnext[idx_ui*npairs_small+idx_ab] = 0.5 * (Jk[idx_fi*npairs_small+idx_ab] +
                                                   Jk[idx_gi*npairs_small+idx_ab]
                                                   - Jk[idx_fg*npairs_small+idx_ab]);
      }
    }
  }
}

/* helper for nj_backprop_sparse; see below */
static inline void nj_backprop_fast_linear_comb(const SparseVector *rf,   // row idx_fi
                                                const SparseVector *rg,   // row idx_gi
                                                const SparseVector *rfg,  // row idx_fg
                                                SparseVector *rout        // row idx_ui (will be overwritten)
                                                ) {
  /* assumes sorted already */
  spvec_zero(rout); /* clear destination */

  /* use direct access to underlying arrays */
  const SparseVectorElement *af = (SparseVectorElement*)rf->elementlist->array;
  const SparseVectorElement *ag = (SparseVectorElement*)rg->elementlist->array;
  const SparseVectorElement *ah = (SparseVectorElement*)rfg->elementlist->array;
  int nf = lst_size(rf->elementlist),
      ng = lst_size(rg->elementlist),
      nh = lst_size(rfg->elementlist);
  int i = 0, j = 0, k = 0;

  /* merge union of indices with 0.5*(f + g - fg) */
  while (i < nf || j < ng || k < nh) {
    int ci = (i<nf) ? af[i].idx : INT_MAX;
    int cj = (j<ng) ? ag[j].idx : INT_MAX;
    int ck = (k<nh) ? ah[k].idx : INT_MAX;
    int c  = (ci<cj ? (ci<ck ? ci:ck) : (cj<ck?cj:ck));

    double vf = (ci==c) ? af[i].val : 0.0;
    double vg = (cj==c) ? ag[j].val : 0.0;
    double vh = (ck==c) ? ah[k].val : 0.0;
    double v  = 0.5 * (vf + vg - vh);

    if (v != 0.0) spvec_set_sorted(rout, c, v);

    if (ci==c) ++i;
    if (cj==c) ++j;
    if (ck==c) ++k;
  }
}

/* version of function above that uses a sparse matrix implementation
   to avoid explosion in size of Jk and Jnext */
void nj_backprop_sparse(SparseMatrix *Jk, SparseMatrix *Jnext, int n, int f, int g, int u,
                        Vector *active) {
  int i;
  int total_nodes = 2*n - 2; /* total possible in final tree */

  /* double buffering Jk and Jnext to avoid expensive copy.  Keep
     track of which destination rows are rebuilt */
  unsigned char *touched = (unsigned char*)calloc(Jk->nrows, 1);
  if (!touched) die("nj_backprop_sparse: out of memory\n");
  
  /* now update distances involving new node u */
  for (i = 0; i < total_nodes; i++) {
    if (vec_get(active, i) == FALSE || i == f || i == g || i == u) continue;  

    int idx_ui = nj_i_j_to_dist(u, i, total_nodes);
    int idx_fi = nj_i_j_to_dist(f, i, total_nodes);
    int idx_gi = nj_i_j_to_dist(g, i, total_nodes);
    int idx_fg = nj_i_j_to_dist(f, g, total_nodes);

    /* ensure destination row is deep copy before writing */
    spmat_replace_row_empty(Jnext, idx_ui, lst_size(Jk->rows[idx_fi]->elementlist) +
                            lst_size(Jk->rows[idx_gi]->elementlist) +
                            lst_size(Jk->rows[idx_fg]->elementlist));
    
    /* use helper function to combine values from three rows in one pass */
    nj_backprop_fast_linear_comb(Jk->rows[idx_fi], Jk->rows[idx_gi],
                                 Jk->rows[idx_fg], Jnext->rows[idx_ui]);

    touched[idx_ui] = 1;
  }

  /* Alias all untouched rows from Jk into Jnext (avoid data copy) */
  for (int r = 0; r < Jk->nrows; r++) {
    if (touched[r] == 1) continue;
    if (Jnext->rows[r] == Jk->rows[r]) continue;  /* already aliased (rare) */
    spvec_release(Jnext->rows[r]);
    Jnext->rows[r] = Jk->rows[r];
    spvec_retain(Jnext->rows[r]);
  }

  free(touched);
}

/* initialize Jk at the beginning of the NJ alg */
void nj_backprop_init(double *Jk, int n) {
  int i, j, total_nodes = 2*n - 2;
  int npairs_large = (total_nodes * (total_nodes - 1)) / 2;
  int npairs_small = (n * (n - 1)) / 2;

  memset(Jk, 0, npairs_large * npairs_small * sizeof(double));
  
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      int idx_ij_row = nj_i_j_to_dist(i, j, total_nodes);
      int idx_ij_col = nj_i_j_to_dist(i, j, n);
      Jk[idx_ij_row*npairs_small+idx_ij_col] = 1;  /* deriv of d_{ij} wrt itself */
    }
  }
}

/* version that uses sparse matrix */
void nj_backprop_init_sparse(SparseMatrix *Jk, int n) {
  int i, j, total_nodes = 2*n - 2;

  spmat_zero(Jk); 
  
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      int idx_ij_row = nj_i_j_to_dist(i, j, total_nodes);
      int idx_ij_col = nj_i_j_to_dist(i, j, n);
      spmat_set_sorted(Jk, idx_ij_row, idx_ij_col, 1.0);  /* deriv of d_{ij} wrt itself */
    }
  }
}

/* for use in backpropation with NJ.  Sets appropriate elements of
   Jacobian dt_dD after two neighbors f and g are joined */
void nj_backprop_set_dt_dD(double *Jk, Matrix *dt_dD, int n, int f, int g,
                           int branch_idx_f, int branch_idx_g, Vector *active) {
  int a, b, m;
  int total_nodes = 2*n - 2;
  int idx_fg = nj_i_j_to_dist(f, g, total_nodes);
  int npairs_small = (n * (n - 1)) / 2;
  int nk = vec_sum(active);

  /* the final call, with nk = 2, is a special case */
  if (nk == 2) {
    /* directly set the final branch derivative */
    for (a = 0; a < n; a++) {
      for (b = a + 1; b < n; b++) {
        int idx_ab = nj_i_j_to_dist(a, b, n);
        
        /* branch derivative is equal to the value in Jk times 1/2
           because of the way we split the last branch in the unrooted
           tree */
        mat_set(dt_dD, branch_idx_f, idx_ab, 0.5 * Jk[idx_fg*npairs_small+idx_ab]);
      }
    }
    return;
  }
    
  /* branch derivative for f -> u */
  for (a = 0; a < n; a++) {
    for (b = a + 1; b < n; b++) {
      int idx_ab = nj_i_j_to_dist(a, b, n);
      double sum_diff = 0;

      for (m = 0; m < total_nodes; m++) {
        if (vec_get(active, m) == FALSE || m == f || m == g)
          continue;

        int idx_fm = nj_i_j_to_dist(f, m, total_nodes);
        int idx_gm = nj_i_j_to_dist(g, m, total_nodes);
        sum_diff += Jk[idx_fm*npairs_small+idx_ab] - Jk[idx_gm*npairs_small+idx_ab];
      }

      mat_set(dt_dD, branch_idx_f, idx_ab, 0.5 * Jk[idx_fg*npairs_small+idx_ab] +
              (0.5 / (nk - 2)) * sum_diff);
      assert(isfinite(mat_get(dt_dD, branch_idx_f, idx_ab)));
    }
  }

  /* branch derivative for g -> u */
  for (a = 0; a < n; a++) {
    for (b = a + 1; b < n; b++) {
      int idx_ab = nj_i_j_to_dist(a, b, n);
      mat_set(dt_dD, branch_idx_g, idx_ab, Jk[idx_fg*npairs_small+idx_ab] -
        mat_get(dt_dD, branch_idx_f, idx_ab));
    }
  }
}

/* version that uses sparse matrix */
void nj_backprop_set_dt_dD_sparse(SparseMatrix *Jk, Matrix *dt_dD, int n, int f, int g,
                                  int branch_idx_f, int branch_idx_g, Vector *active) {
  int total_nodes = 2*n - 2;
  int idx_fg = nj_i_j_to_dist(f, g, total_nodes);
  int nk = vec_sum(active);
  int n_ab = n*(n-1)/2;
  double *sum_diff = malloc(sizeof(double) * n_ab);

  /* the final call, with nk = 2, is a special case */
  if (nk == 2) {
    /* just copy 0.5 * row idx_fg into branch f */
    const SparseVector *rfg = Jk->rows[idx_fg];

    /* first zero the whole dt_dD row */
    for (int ab = 0; ab < n_ab; ab++) mat_set(dt_dD, branch_idx_f, ab, 0.0);
    const SparseVectorElement *a = (SparseVectorElement*)rfg->elementlist->array;
    int nz = lst_size(rfg->elementlist);
    /* now fill in non-zero entries */
    for (int t = 0; t < nz; t++)
      mat_set(dt_dD, branch_idx_f, a[t].idx, 0.5 * a[t].val);

    free(sum_diff);
    return;
  }

  for (int ab = 0; ab < n_ab; ab++) sum_diff[ab] = 0.0;
  
  /* for each active m (excluding f,g), accumulate sparse diffs */
  for (int m = 0; m < total_nodes; m++) {
    if (vec_get(active, m) == FALSE || m == f || m == g)
      continue;

    int idx_fm = nj_i_j_to_dist(f, m, total_nodes);
    int idx_gm = nj_i_j_to_dist(g, m, total_nodes);

    const SparseVectorElement *rf = (SparseVectorElement*)Jk->rows[idx_fm]->elementlist->array;
    const SparseVectorElement *rg = (SparseVectorElement*)Jk->rows[idx_gm]->elementlist->array;
    int nf = lst_size(Jk->rows[idx_fm]->elementlist);
    int ng = lst_size(Jk->rows[idx_gm]->elementlist);
    int i = 0, j = 0;

    /* branch derivative for f->u; first merge the two rows then
       scatter (+1 for f, -1 for g) */
    while (i < nf || j < ng) {
      int cf = (i<nf) ? rf[i].idx : INT_MAX;
      int cg = (j<ng) ? rg[j].idx : INT_MAX;
      if (cf == cg) {
        sum_diff[cf] += (rf[i].val - rg[j].val);
        i++; j++;
      }
      else if (cf < cg) {
        sum_diff[cf] += rf[i].val;
        i++;
      }
      else {
        sum_diff[cg] -= rg[j].val;
        j++;
      }
    }
  }

  /* set dt_dD row for branch f: 0.5*Jk[idx_fg,:] + (0.5/(nk-2))*sum_diff */
  const SparseVectorElement *ah = (SparseVectorElement*)Jk->rows[idx_fg]->elementlist->array;
  int nh = lst_size(Jk->rows[idx_fg]->elementlist);

  /* start from (0.5/(nk-2))*sum_diff (dense) */
  double scale = 0.5 / (nk - 2);
  for (int ab = 0; ab < n_ab; ab++)
    mat_set(dt_dD, branch_idx_f, ab, scale * sum_diff[ab]);

  /* now add the sparse 0.5 * Jk[idx_fg,:] */
  for (int t = 0; t < nh; t++) {
    int ab = ah[t].idx;
    mat_set(dt_dD, branch_idx_f, ab,
            mat_get(dt_dD, branch_idx_f, ab) + 0.5 * ah[t].val);
  }

  /* branch derivative for g->u; g = Jk[idx_fg,:] - branch f 
     do it densely; add sparse afterwards */
  for (int ab = 0; ab < n_ab; ab++) {
    double jf = mat_get(dt_dD, branch_idx_f, ab);
    mat_set(dt_dD, branch_idx_g, ab, -jf); /* temporarily 0, will add Jk[idx_fg,:] */ 
  }

  for (int t = 0; t < nh; t++) {
    int ab = ah[t].idx;
    mat_set(dt_dD, branch_idx_g, ab,
            mat_get(dt_dD, branch_idx_g, ab) + ah[t].val);
  }
  
  free(sum_diff);
}

