/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* calculation of robinson foulds distances */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>
#include "phast/stacks.h"
#include "phast/trees.h"
#include "phast/misc.h"
#include "phast/stringsplus.h"
#include "phast/hashtable.h"
#include "rf.h"

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

/* allocate zeroed mask with W words */
static BitMask *bm_new(int W) {
  BitMask *m = smalloc(sizeof(BitMask));
  m->W = W;
  m->w = calloc(W, sizeof(uint64_t));
  return m;
}

static void bm_free(BitMask *m) { if (!m) return; free(m->w); sfree(m); }

/* set a single bit (0-based global bit index) */
static inline void bm_set(BitMask *m, int bit) {
  int wi = bit >> 6, bi = bit & 63;
  m->w[wi] |= (uint64_t)1ULL << bi;
}

/* dst = ~src; then clear high unused bits above nbits */
static inline void bm_not(BitMask *dst, const BitMask *src, int nbits) {
  for (int i = 0; i < dst->W; ++i) dst->w[i] = ~src->w[i];
  int rem = nbits & 63;
  if (rem) {
    uint64_t keep = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
    dst->w[dst->W - 1] &= keep;
  }
}

/* count set bits */
static inline int bm_popcount(const BitMask *m) {
  int s = 0; for (int i = 0; i < m->W; ++i) s += popcount64(m->w[i]); return s;
}

/* lexicographic compare (lowest word first is fine as long as consistent) */
int tr_bitmask_cmp(const BitMask *a, const BitMask *b) {
  /* compare from high word to low word for nice order */
  for (int i = a->W - 1; i >= 0; --i) {
    if (a->w[i] < b->w[i]) return -1;
    if (a->w[i] > b->w[i]) return 1;
  }
  return 0;
}

static int bm_cmp_words(const void *pa, const void *pb) {
  const BitMask *a = *(BitMask* const*)pa;
  const BitMask *b = *(BitMask* const*)pb;
  return tr_bitmask_cmp(a, b);
}

void tr_bitmask_free(BitMask *m) { bm_free(m); }

/* deep copy */
static BitMask *bm_clone(const BitMask *m) {
  BitMask *c = bm_new(m->W);
  memcpy(c->w, m->w, sizeof(uint64_t)*m->W);
  return c;
}

/* canonicalize split (make it the smaller side). Returns NEW mask on heap.
   For balanced splits (sz == other, only possible when nbits is even)
   tie-break by keeping the half that contains taxon 0 so the same
   split always maps to the same mask regardless of traversal order. */
static BitMask *bm_canonical(const BitMask *m, int nbits) {
  int sz = bm_popcount(m);
  int other = nbits - sz;
  if (sz == 0 || other == 0) return NULL;            /* trivial */
  if (sz == 1 || other == 1) return NULL;            /* leaf edge; ignore */

  int keep_m;
  if (sz < other) keep_m = 1;
  else if (sz > other) keep_m = 0;
  else /* balanced */ keep_m = (m->w[0] & 1ULL) ? 1 : 0;

  if (keep_m) {
    return bm_clone(m);
  } else {
    BitMask *c = bm_new(m->W);
    bm_not(c, m, nbits);
    return c;
  }
}

static void mv_init(MaskVec *mv, int cap) {
  mv->cap = cap > 0 ? cap : 16; mv->size = 0;
  mv->a = smalloc(sizeof(BitMask*) * mv->cap);
}
static void mv_push(MaskVec *mv, BitMask *m) {
  if (m == NULL) return; /* trivial split filtered */
  if (mv->size == mv->cap) {
    mv->cap *= 2;
    mv->a = srealloc(mv->a, sizeof(BitMask*) * mv->cap);
  }
  mv->a[mv->size++] = m;
}
static void mv_free(MaskVec *mv) {
  for (int i = 0; i < mv->size; ++i) bm_free(mv->a[i]);
  sfree(mv->a); mv->a = NULL; mv->size = mv->cap = 0;
}

static inline int name_to_index(Hashtable *ht, const char *name) {
  void *vp = hsh_get(ht, name);
  if (vp == (void*)-1) return -1;
  return ptr_to_int(vp);
}

/* DFS over TreeNode to collect edge splits */
/* Returns heap mask for subtree of u. Parent is used to avoid back-edge in unrooted reps. */
static BitMask *dfs_collect_splits(TreeNode *u, TreeNode *parent,
                                   Hashtable *name2idx, int n, int W,
                                   MaskVec *splits) {
  if (u->lchild == NULL && u->rchild == NULL) {
    /* leaf */
    int idx = name_to_index(name2idx, u->name);
    if (idx < 0) die("Leaf '%s' not in name list.\n", u->name);
    BitMask *m = bm_new(W);
    bm_set(m, idx);
    return m;
  }

  BitMask *mask_u = bm_new(W);

  /* left child */
  if (u->lchild && u->lchild != parent) {
    BitMask *ml = dfs_collect_splits(u->lchild, u, name2idx, n, W, splits);
    /* edge split defined by (u—lchild) => use ml as one side */
    BitMask *canL = bm_canonical(ml, n);
    mv_push(splits, canL);
    /* accumulate */
    for (int i = 0; i < W; ++i) mask_u->w[i] |= ml->w[i];
    bm_free(ml);
  }
  /* right child */
  if (u->rchild && u->rchild != parent) {
    BitMask *mr = dfs_collect_splits(u->rchild, u, name2idx, n, W, splits);
    BitMask *canR = bm_canonical(mr, n);
    mv_push(splits, canR);
    for (int i = 0; i < W; ++i) mask_u->w[i] |= mr->w[i];
    bm_free(mr);
  }

  /* If your TreeNode can have >2 children (multifurcating),
     iterate a generic adjacency list here; the logic is the same:
     for each child c!=parent, collect mc, push canonical(mc), OR into mask_u. */

  return mask_u;
}

/* calculate the symmetric Robinson Foulds distance between two
   trees. Considers topology only.  Leaf names must match exactly.
   This is an O(n log n) implementation  */
double tr_robinson_foulds(TreeNode *t1, TreeNode *t2) {
  /* build sorted lists of leaf names */
  List *names1 = tr_leaf_names(t1);
  List *names2 = tr_leaf_names(t2);
  lst_qsort_str(names1, ASCENDING);
  lst_qsort_str(names2, ASCENDING);
  if (str_list_equal(names1, names2) == FALSE)
    die("ERROR in tr_robinson_foulds: trees do not have matching leaf names.\n");

  int n = lst_size(names1);
  if (n < 3) { lst_free_strings(names1); lst_free_strings(names2); return 0; }

  /* build name2idx map from names1 */
  Hashtable *name2idx = hsh_new(2 * lst_size(names1)); 
  for (int i = 0; i < lst_size(names1); i++) {
    String *s = lst_get_ptr(names1, i);
    hsh_put_int(name2idx, s->chars, i);
  }
  
  const int W = (n + 63) >> 6; /* words needed */

  /* collect canonical splits from each tree */
  MaskVec S1, S2;
  mv_init(&S1, n); mv_init(&S2, n);

  BitMask *rootmask1 = dfs_collect_splits(t1, NULL, name2idx, n, W, &S1);
  BitMask *rootmask2 = dfs_collect_splits(t2, NULL, name2idx, n, W, &S2);
  bm_free(rootmask1); bm_free(rootmask2);

  /* sort both split multisets (lexicographic) */
  qsort(S1.a, S1.size, sizeof(BitMask*), bm_cmp_words);
  qsort(S2.a, S2.size, sizeof(BitMask*), bm_cmp_words);

  /* count common splits (intersection) with two pointers */
  int i = 0, j = 0, common = 0;
  while (i < S1.size && j < S2.size) {
    int cmp = bm_cmp_words(&S1.a[i], &S2.a[j]);
    if (cmp == 0) { common++; i++; j++; }
    else if (cmp < 0) i++;
    else j++;
  }

  /* symmetric RF distance = |S1| + |S2| - 2|common| */
  int RF = (S1.size + S2.size) - 2*common;

  /* cleanup */
  mv_free(&S1); mv_free(&S2);
  lst_free_strings(names2);
  lst_free_strings(names1);
  hsh_free(name2idx);

  return RF;
}

/* ---- helpers for tr_tree_entropy ---------------------------------------- */

/* Like bm_canonical but does NOT filter trivial (leaf) splits, so all 2n-3
   edges of a bifurcating tree are included. */
static BitMask *bm_canonical_any(const BitMask *m, int nbits) {
  int sz = bm_popcount(m);
  int other = nbits - sz;
  if (sz == 0 || other == 0) return NULL;
  if (sz <= other)
    return bm_clone(m);
  BitMask *c = bm_new(m->W);
  bm_not(c, m, nbits);
  return c;
}

/* (canonical split, branch length) pair. */
typedef struct { BitMask *mask; double blen; } SplitBLen;

/* qsort comparator for SplitBLen by mask. */
static int sbl_cmp(const void *pa, const void *pb) {
  return bm_cmp_words(&((const SplitBLen*)pa)->mask,
                      &((const SplitBLen*)pb)->mask);
}

/* DFS that collects ALL edges (including leaf edges), writing (canonical
   split, branch length) pairs into out[].  Returns the subtree BitMask for
   node u (caller must bm_free it).
   is_root should be TRUE only for the initial (top-level) call.  When the
   root has exactly two children -- the usual case for a Newick file
   representing what is really an unrooted tree -- those two child edges
   are the SAME true bipartition (each child's subtree is exactly the
   complement of the other), so only one of them is recorded.  Without this,
   a bifurcating-root tree yields 2n-2 edges instead of the correct 2n-3
   for an unrooted binary tree, double-counting one split per tree. */
static BitMask *dfs_splits_blen_r(TreeNode *u, TreeNode *parent, int is_root,
                                  Hashtable *name2idx, int n, int W,
                                  SplitBLen *out, int *nout) {
  if (u->lchild == NULL && u->rchild == NULL) {
    int idx = name_to_index(name2idx, u->name);
    if (idx < 0) die("Leaf '%s' not in name list.\n", u->name);
    BitMask *m = bm_new(W);
    bm_set(m, idx);
    return m;
  }
  BitMask *mask_u = bm_new(W);
  TreeNode *ch[2] = {u->lchild, u->rchild};
  int nchildren = (ch[0] && ch[0] != parent) + (ch[1] && ch[1] != parent);
  int seen = 0;
  for (int ci = 0; ci < 2; ci++) {
    TreeNode *c = ch[ci];
    if (!c || c == parent) continue;
    BitMask *mc = dfs_splits_blen_r(c, u, FALSE, name2idx, n, W, out, nout);
    BitMask *can = bm_canonical_any(mc, n);
    int skip_dup_root_edge = (is_root && nchildren == 2 && seen == 1);
    if (can && !skip_dup_root_edge) {
      out[*nout].mask = can;
      out[*nout].blen = c->dparent > 0.0 ? c->dparent : 1e-300;
      (*nout)++;
    }
    else if (can) {
      bm_free(can);
    }
    for (int i = 0; i < W; i++) mask_u->w[i] |= mc->w[i];
    bm_free(mc);
    seen++;
  }
  return mask_u;
}

static BitMask *dfs_splits_blen(TreeNode *u, TreeNode *parent,
                                Hashtable *name2idx, int n, int W,
                                SplitBLen *out, int *nout) {
  return dfs_splits_blen_r(u, parent, TRUE, name2idx, n, W, out, nout);
}

/* Per-tree data: sorted (canonical split, branch length) array. */
typedef struct { SplitBLen *sbl; int m; } TreeSplitData;

/* Global state for qsort topology comparison (C89-style callback). */
static TreeSplitData *g_tsd;

static int cmp_tsd_idx(const void *pa, const void *pb) {
  const TreeSplitData *a = &g_tsd[*(const int*)pa];
  const TreeSplitData *b = &g_tsd[*(const int*)pb];
  int m = a->m < b->m ? a->m : b->m;
  for (int k = 0; k < m; k++) {
    int c = bm_cmp_words(&a->sbl[k].mask, &b->sbl[k].mask);
    if (c != 0) return c;
  }
  return a->m - b->m;
}

static int tsd_same_topo(const TreeSplitData *a, const TreeSplitData *b) {
  if (a->m != b->m) return 0;
  for (int k = 0; k < a->m; k++)
    if (bm_cmp_words(&a->sbl[k].mask, &b->sbl[k].mask) != 0) return 0;
  return 1;
}

/* ---- public function ----------------------------------------------------- */

/* Compute split entropy, topology entropy, and mean branch-length variance
 * for a collection of trees.
 *
 * H_split:  sum of Bernoulli entropies over all non-trivial splits.
 *
 * H_top:    Shannon entropy over distinct topologies, -Σ p(τ) log p(τ).
 *
 * mean_var: topology-frequency-weighted sum of sample variances of log
 *   branch lengths, summed over branches.  For topology τ with n_τ
 *   samples, the per-branch variance is σ²_τk for branch k.
 *   Topology groups with n_τ = 1 contribute zero.
 *
 * mean_var_per_branch: mean_var / m  (m = number of branches per tree).
 */
void tr_tree_entropy(List *trees, double *H_split, double *H_top,
                     double *mean_var, double *mean_var_per_branch) {
  int S = lst_size(trees);
  *H_split             = 0.0;
  *H_top               = 0.0;
  *mean_var            = 0.0;
  *mean_var_per_branch = 0.0;
  if (S == 0) return;

  TreeNode *t0 = lst_get_ptr(trees, 0);
  List *names = tr_leaf_names(t0);
  lst_qsort_str(names, ASCENDING);
  int n = lst_size(names);

  if (n < 3) {
    lst_free_strings(names);
    lst_free(names);
    return;  /* all three remain 0 */
  }

  int W = (n + 63) >> 6;
  int max_edges = 2 * n;  /* generous upper bound */

  Hashtable *name2idx = hsh_new(2 * n);
  for (int i = 0; i < n; i++) {
    String *s = lst_get_ptr(names, i);
    hsh_put_int(name2idx, s->chars, i);
  }

  /* ---- collect per-tree (split, blen) arrays ---- */
  TreeSplitData *tsd = smalloc(S * sizeof(TreeSplitData));
  MaskVec nontrivial;  /* flat pool of non-trivial splits for H_split */
  mv_init(&nontrivial, S * (n - 3 > 0 ? n - 3 : 1));

  for (int s = 0; s < S; s++) {
    TreeNode *t = lst_get_ptr(trees, s);
    SplitBLen *sbl = smalloc(max_edges * sizeof(SplitBLen));
    int nbl = 0;
    BitMask *root = dfs_splits_blen(t, NULL, name2idx, n, W, sbl, &nbl);
    bm_free(root);
    qsort(sbl, nbl, sizeof(SplitBLen), sbl_cmp);
    tsd[s].sbl = sbl;
    tsd[s].m   = nbl;
    /* gather non-trivial splits for H_split */
    for (int k = 0; k < nbl; k++) {
      int sz = bm_popcount(sbl[k].mask);
      if (sz >= 2 && (n - sz) >= 2)
        mv_push(&nontrivial, bm_clone(sbl[k].mask));
    }
  }

  /* ---- H_split ---- */
  qsort(nontrivial.a, nontrivial.size, sizeof(BitMask*), bm_cmp_words);
  {
    int i = 0;
    while (i < nontrivial.size) {
      int j = i + 1;
      while (j < nontrivial.size &&
             bm_cmp_words(&nontrivial.a[i], &nontrivial.a[j]) == 0)
        j++;
      double p = (double)(j - i) / S;
      if (p > 0.0 && p < 1.0)
        *H_split += -p * log(p) - (1.0 - p) * log(1.0 - p);
      i = j;
    }
  }
  mv_free(&nontrivial);

  /* ---- H_branch: group trees by topology, compute sample covariance ---- */
  int *order = smalloc(S * sizeof(int));
  for (int s = 0; s < S; s++) order[s] = s;
  g_tsd = tsd;
  qsort(order, S, sizeof(int), cmp_tsd_idx);

  int gi = 0;
  while (gi < S) {
    int gj = gi + 1;
    while (gj < S && tsd_same_topo(&tsd[order[gi]], &tsd[order[gj]]))
      gj++;
    int ntau = gj - gi;
    int m    = tsd[order[gi]].m;
    double p_tau = (double)ntau / S;
    *H_top += -p_tau * log(p_tau);

    if (ntau >= 2 && m >= 1) {
      /* sample mean of log branch lengths */
      double *mu = calloc(m, sizeof(double));
      for (int si = gi; si < gj; si++) {
        SplitBLen *sbl = tsd[order[si]].sbl;
        for (int k = 0; k < m; k++)
          mu[k] += log(sbl[k].blen);
      }
      for (int k = 0; k < m; k++) mu[k] /= ntau;

      /* weighted sum of sample variances over branches */
      double lv = 0.0;
      for (int k = 0; k < m; k++) {
        double var_k = 0.0;
        for (int si = gi; si < gj; si++) {
          double d = log(tsd[order[si]].sbl[k].blen) - mu[k];
          var_k += d * d;
        }
        var_k /= (ntau - 1);
        lv += var_k;
      }
      *mean_var += p_tau * lv;
      free(mu);
    }
    gi = gj;
  }
  /* per-branch: divide by number of branches (same for all trees) */
  int m_all = tsd[order[0]].m;
  *mean_var_per_branch = (m_all > 0) ? *mean_var / m_all : 0.0;
  sfree(order);

  /* ---- cleanup ---- */
  for (int s = 0; s < S; s++) {
    for (int k = 0; k < tsd[s].m; k++) bm_free(tsd[s].sbl[k].mask);
    sfree(tsd[s].sbl);
  }
  sfree(tsd);
  lst_free_strings(names);
  lst_free(names);
  hsh_free(name2idx);
}

/* ---- shared split-frequency helper (used by kl.c's tr_split_kl) ---------- */

/* Collect unique non-trivial splits and their occurrence counts across a
 * set of trees.  Returns a newly allocated array sorted by mask, with
 * *nsc_out set to its length.  *nleaves_out (if non-NULL) is set to the
 * number of leaves.  Returns NULL (with *nsc_out == 0) if trees is empty
 * or has fewer than 3 leaves.  Caller frees each element's mask via
 * tr_bitmask_free, then frees the returned array itself. */
SplitCount *tr_collect_split_counts(List *trees, int *nsc_out, int *nleaves_out) {
  *nsc_out = 0;
  int S = lst_size(trees);
  if (S == 0) { if (nleaves_out) *nleaves_out = 0; return NULL; }

  TreeNode *t0 = lst_get_ptr(trees, 0);
  List *names = tr_leaf_names(t0);
  lst_qsort_str(names, ASCENDING);
  int n = lst_size(names);
  if (nleaves_out) *nleaves_out = n;
  if (n < 3) { lst_free_strings(names); lst_free(names); return NULL; }

  int W = (n + 63) >> 6;
  int max_edges = 2 * n;
  Hashtable *name2idx = hsh_new(2 * n);
  for (int i = 0; i < n; i++) {
    String *s = lst_get_ptr(names, i);
    hsh_put_int(name2idx, s->chars, i);
  }

  MaskVec all;
  mv_init(&all, S * (n - 3 > 0 ? n - 3 : 1));
  for (int s = 0; s < S; s++) {
    TreeNode *t = lst_get_ptr(trees, s);
    SplitBLen *sbl = smalloc(max_edges * sizeof(SplitBLen));
    int nbl = 0;
    BitMask *root = dfs_splits_blen(t, NULL, name2idx, n, W, sbl, &nbl);
    bm_free(root);
    for (int k = 0; k < nbl; k++) {
      int sz = bm_popcount(sbl[k].mask);
      if (sz >= 2 && (n - sz) >= 2)
        mv_push(&all, bm_clone(sbl[k].mask));
      bm_free(sbl[k].mask);
    }
    sfree(sbl);
  }
  qsort(all.a, all.size, sizeof(BitMask*), bm_cmp_words);

  SplitCount *out = smalloc((all.size > 0 ? all.size : 1) * sizeof(SplitCount));
  int nsc = 0, i = 0;
  while (i < all.size) {
    int j = i + 1;
    while (j < all.size && bm_cmp_words(&all.a[i], &all.a[j]) == 0) j++;
    out[nsc].mask = bm_clone(all.a[i]);
    out[nsc].count = j - i;
    nsc++;
    i = j;
  }
  mv_free(&all);
  lst_free_strings(names);
  lst_free(names);
  hsh_free(name2idx);

  *nsc_out = nsc;
  return out;
}

