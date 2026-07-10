/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Branch-score (Kuhner-Felsenstein) distance */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "phast/trees.h"
#include "phast/misc.h"
#include "phast/stringsplus.h"
#include "phast/hashtable.h"
#include "branch_score.h"

/* bitset for up to many thousands of leaves */
typedef struct {
  int W;            /* number of 64-bit words */
  uint64_t *w;      /* words */
} BitMask;

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
static int bm_cmp_words(const void *pa, const void *pb) {
  const BitMask *a = *(BitMask* const*)pa;
  const BitMask *b = *(BitMask* const*)pb;
  /* compare from high word to low word for nice order */
  for (int i = a->W - 1; i >= 0; --i) {
    if (a->w[i] < b->w[i]) return -1;
    if (a->w[i] > b->w[i]) return 1;
  }
  return 0;
}

/* deep copy */
static BitMask *bm_clone(const BitMask *m) {
  BitMask *c = bm_new(m->W);
  memcpy(c->w, m->w, sizeof(uint64_t)*m->W);
  return c;
}

static inline int name_to_index(Hashtable *ht, const char *name) {
  void *vp = hsh_get(ht, name);
  if (vp == (void*)-1) return -1;
  return ptr_to_int(vp);
}

/* (canonical split, branch length) pair. */
typedef struct { BitMask *mask; double blen; } SplitBLen;

/* qsort comparator for SplitBLen by mask. */
static int sbl_cmp(const void *pa, const void *pb) {
  return bm_cmp_words(&((const SplitBLen*)pa)->mask,
                      &((const SplitBLen*)pb)->mask);
}

/* Canonicalize a split, KEEPING leaf edges
 * and tie-breaking balanced splits by the side containing taxon 0, so the
 * same unrooted split maps to the same mask across trees regardless of
 * rooting/traversal.  Returns NULL only for the trivial whole-set / empty
 * split. */
static BitMask *bm_canonical_all(const BitMask *m, int nbits) {
  int sz = bm_popcount(m);
  int other = nbits - sz;
  if (sz == 0 || other == 0) return NULL;
  int keep_m;
  if (sz < other) keep_m = 1;
  else if (sz > other) keep_m = 0;
  else keep_m = (m->w[0] & 1ULL) ? 1 : 0;   /* balanced: keep taxon-0 side */
  if (keep_m) return bm_clone(m);
  BitMask *c = bm_new(m->W);
  bm_not(c, m, nbits);
  return c;
}

/* Collect all splits, including leaf edges. */
static BitMask *dfs_splits_blen_all(TreeNode *u, TreeNode *parent,
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
  for (int ci = 0; ci < 2; ci++) {
    TreeNode *c = ch[ci];
    if (!c || c == parent) continue;
    BitMask *mc = dfs_splits_blen_all(c, u, name2idx, n, W, out, nout);
    BitMask *can = bm_canonical_all(mc, n);
    if (can) {
      out[*nout].mask = can;
      out[*nout].blen = c->dparent > 0.0 ? c->dparent : 0.0;
      (*nout)++;
    }
    for (int i = 0; i < W; i++) mask_u->w[i] |= mc->w[i];
    bm_free(mc);
  }
  return mask_u;
}

/* Build name->index hashtable from a tree's sorted leaf names. */
static Hashtable *bs_name2idx(TreeNode *t, int *n_out, int *W_out) {
  List *names = tr_leaf_names(t);
  lst_qsort_str(names, ASCENDING);
  int n = lst_size(names);
  Hashtable *h = hsh_new(2 * (n > 0 ? n : 2));
  for (int i = 0; i < n; i++) {
    String *s = lst_get_ptr(names, i);
    hsh_put_int(h, s->chars, i);
  }
  lst_free_strings(names);
  lst_free(names);
  *n_out = n;
  *W_out = (n + 63) >> 6;
  return h;
}

/* Collect a tree's unrooted split->length vector: sorted by canonical split,
 * with duplicate canonical splits merged by summing lengths (this reconstructs
 * the single unrooted edge length from a rooted tree's two root half-edges).
 * Caller frees via bs_free_vector. */
static SplitBLen *bs_tree_vector(TreeNode *t, Hashtable *name2idx,
                                 int n, int W, int *m_out) {
  int max_edges = 2 * n + 2;
  SplitBLen *sbl = smalloc(max_edges * sizeof(SplitBLen));
  int nbl = 0;
  BitMask *root = dfs_splits_blen_all(t, NULL, name2idx, n, W, sbl, &nbl);
  bm_free(root);
  qsort(sbl, nbl, sizeof(SplitBLen), sbl_cmp);
  int w = 0;
  for (int r = 0; r < nbl; ) {
    int s = r + 1;
    double sum = sbl[r].blen;
    while (s < nbl && bm_cmp_words(&sbl[r].mask, &sbl[s].mask) == 0) {
      sum += sbl[s].blen;
      bm_free(sbl[s].mask);
      s++;
    }
    sbl[w].mask = sbl[r].mask;
    sbl[w].blen = sum;
    w++;
    r = s;
  }
  *m_out = w;
  return sbl;
}

static void bs_free_vector(SplitBLen *sbl, int m) {
  for (int k = 0; k < m; k++) bm_free(sbl[k].mask);
  sfree(sbl);
}

/* Euclidean distance between two sorted split->length vectors. */
static double bs_dist_sorted(const SplitBLen *a, int ma,
                             const SplitBLen *b, int mb) {
  double s = 0.0;
  int i = 0, j = 0;
  while (i < ma && j < mb) {
    int c = bm_cmp_words(&a[i].mask, &b[j].mask);
    if (c == 0) { double d = a[i].blen - b[j].blen; s += d * d; i++; j++; }
    else if (c < 0) { s += a[i].blen * a[i].blen; i++; }
    else { s += b[j].blen * b[j].blen; j++; }
  }
  for (; i < ma; i++) s += a[i].blen * a[i].blen;
  for (; j < mb; j++) s += b[j].blen * b[j].blen;
  return sqrt(s);
}

double tr_branch_score(TreeNode *t1, TreeNode *t2) {
  /* verify matching leaf sets */
  List *n1 = tr_leaf_names(t1), *n2 = tr_leaf_names(t2);
  lst_qsort_str(n1, ASCENDING);
  lst_qsort_str(n2, ASCENDING);
  if (str_list_equal(n1, n2) == FALSE)
    die("ERROR in tr_branch_score: trees do not have matching leaf names.\n");
  lst_free_strings(n1); lst_free(n1);
  lst_free_strings(n2); lst_free(n2);

  int n, W;
  Hashtable *name2idx = bs_name2idx(t1, &n, &W);
  if (n < 2) { hsh_free(name2idx); return 0.0; }
  int m1, m2;
  SplitBLen *v1 = bs_tree_vector(t1, name2idx, n, W, &m1);
  SplitBLen *v2 = bs_tree_vector(t2, name2idx, n, W, &m2);
  double d = bs_dist_sorted(v1, m1, v2, m2);
  bs_free_vector(v1, m1);
  bs_free_vector(v2, m2);
  hsh_free(name2idx);
  return d;
}

double tr_branch_score_pointest(List *trees, TreeNode *ref) {
  int S = lst_size(trees);
  if (S == 0) return 0.0;
  int n, W;
  Hashtable *name2idx = bs_name2idx(ref, &n, &W);
  if (n < 2) { hsh_free(name2idx); return 0.0; }

  /* gather all (split, blen) pairs from all trees into one array */
  int cap = 16, na = 0;
  SplitBLen *A = smalloc(cap * sizeof(SplitBLen));
  for (int s = 0; s < S; s++) {
    TreeNode *t = lst_get_ptr(trees, s);
    int mt;
    SplitBLen *vt = bs_tree_vector(t, name2idx, n, W, &mt);
    if (na + mt > cap) {
      while (na + mt > cap) cap *= 2;
      A = srealloc(A, cap * sizeof(SplitBLen));
    }
    for (int k = 0; k < mt; k++) A[na++] = vt[k];  /* transfer mask ownership */
    sfree(vt);
  }
  qsort(A, na, sizeof(SplitBLen), sbl_cmp);

  /* collapse to the posterior-mean vector: mean length = sum over trees / S
     (splits absent from a tree contribute length 0, hence divide by S) */
  int mm = 0;
  SplitBLen *M = smalloc((na > 0 ? na : 1) * sizeof(SplitBLen));
  for (int r = 0; r < na; ) {
    int s = r + 1;
    double sum = A[r].blen;
    while (s < na && bm_cmp_words(&A[r].mask, &A[s].mask) == 0) {
      sum += A[s].blen;
      bm_free(A[s].mask);
      s++;
    }
    M[mm].mask = A[r].mask;
    M[mm].blen = sum / S;
    mm++;
    r = s;
  }
  sfree(A);

  int mref;
  SplitBLen *R = bs_tree_vector(ref, name2idx, n, W, &mref);
  double d = bs_dist_sorted(M, mm, R, mref);
  bs_free_vector(M, mm);
  bs_free_vector(R, mref);
  hsh_free(name2idx);
  return d;
}

static double bs_subtree_len(TreeNode *u, TreeNode *parent) {
  double s = 0.0;
  if (parent != NULL) s += (u->dparent > 0.0 ? u->dparent : 0.0);
  if (u->lchild && u->lchild != parent) s += bs_subtree_len(u->lchild, u);
  if (u->rchild && u->rchild != parent) s += bs_subtree_len(u->rchild, u);
  return s;
}

double tr_tree_length(TreeNode *t) { return bs_subtree_len(t, NULL); }

