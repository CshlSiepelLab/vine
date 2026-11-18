/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* Local posterior approximation around a point-estimate tree
   (--posterior); see local_posterior.h for an overview.

   The point-estimate tree is represented by child/parent arrays
   indexed by node id, which makes NNI moves cheap.  Each unrooted
   branch is identified with the node below it, except that the two
   branches below the (degree-2) root together form a single unrooted
   branch; its length is varied through the left child of the root
   only, with the right child's length held fixed.

   Procedure:
   1. Diagonal Newton iterations on the branch lengths, starting from
      the point estimate, using the analytical gradient and a
      finite-difference curvature for each branch.
   2. Curvature recomputed at the optimized branch lengths, giving a
      Laplace approximation N(t_hat, 1/h) truncated at zero for each
      branch (with the gradient retained, so branches at the zero
      boundary get the appropriate truncated distribution).
   3. Each internal branch is scored against its two NNI alternatives,
      holding the other branch lengths fixed; the probability of each
      configuration is proportional to its likelihood.
   4. Each sampled tree applies independently drawn NNI configurations
      in random order, skipping any move adjacent to one already
      applied in that sample, and draws each branch length from its
      Laplace approximation. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <phast/misc.h>
#include <local_posterior.h>
#include <likelihoods.h>
#include <crispr.h>
#include <migration.h>

typedef struct {
  int nn, n, root;
  int *lc, *rc, *par;   /* -1 where absent */
  int *leafidx;         /* index into names for leaves, else -1 */
} LpTree;

typedef struct {
  int v, p;     /* endpoints; for the root branch, the two root children */
  int isroot;
} LpEdge;

/* uniform on the open interval (0,1) */
static double lp_unif(void) {
  return ((double)random() + 0.5) / ((double)RAND_MAX + 1.0);
}

static double lp_norm(void) {
  return sqrt(-2.0 * log(lp_unif())) * cos(2.0 * M_PI * lp_unif());
}

/* draw from N(mean, sd^2) truncated to [0, inf) */
static double lp_trunc_norm(double mean, double sd) {
  double a = -mean / sd, z;   /* standardized lower bound */
  if (a <= 0) {               /* bound below the mode: simple rejection */
    do z = lp_norm(); while (z < a);
  }
  else {                      /* tail: exponential rejection (Robert 1995) */
    double alpha = 0.5 * (a + sqrt(a * a + 4.0));
    do z = a - log(lp_unif()) / alpha;
    while (lp_unif() > exp(-0.5 * (z - alpha) * (z - alpha)));
  }
  return mean + sd * z;
}

static LpTree *lp_tree_new(TreeNode *t, char **names, int n) {
  LpTree *T = smalloc(sizeof(LpTree));
  int i, j;

  T->nn = t->nnodes; T->n = n; T->root = t->id;
  T->lc = smalloc(T->nn * sizeof(int));
  T->rc = smalloc(T->nn * sizeof(int));
  T->par = smalloc(T->nn * sizeof(int));
  T->leafidx = smalloc(T->nn * sizeof(int));

  for (i = 0; i < T->nn; i++) {
    TreeNode *nd = lst_get_ptr(t->nodes, i);
    if ((nd->lchild == NULL) != (nd->rchild == NULL))
      die("ERROR in lp_tree_new: tree must be binary.\n");
    T->lc[nd->id] = nd->lchild ? nd->lchild->id : -1;
    T->rc[nd->id] = nd->rchild ? nd->rchild->id : -1;
    T->par[nd->id] = nd->parent ? nd->parent->id : -1;
    T->leafidx[nd->id] = -1;
    if (nd->lchild == NULL) {
      for (j = 0; j < n; j++)
        if (strcmp(nd->name, names[j]) == 0) break;
      if (j == n)
        die("ERROR in lp_tree_new: leaf %s not found.\n", nd->name);
      T->leafidx[nd->id] = j;
    }
  }
  return T;
}

/* copy of topology arrays; leafidx is shared (read only) */
static LpTree *lp_tree_copy(LpTree *S) {
  LpTree *T = smalloc(sizeof(LpTree));
  *T = *S;
  T->lc = smalloc(S->nn * sizeof(int));
  T->rc = smalloc(S->nn * sizeof(int));
  T->par = smalloc(S->nn * sizeof(int));
  memcpy(T->lc, S->lc, S->nn * sizeof(int));
  memcpy(T->rc, S->rc, S->nn * sizeof(int));
  memcpy(T->par, S->par, S->nn * sizeof(int));
  return T;
}

static void lp_tree_free(LpTree *T, int free_leafidx) {
  sfree(T->lc); sfree(T->rc); sfree(T->par);
  if (free_leafidx) sfree(T->leafidx);
  sfree(T);
}

/* exchange the subtrees rooted at x and y (which must have different
   parents) */
static void lp_swap(LpTree *T, int x, int y) {
  int px = T->par[x], py = T->par[y];
  if (T->lc[px] == x) T->lc[px] = y; else T->rc[px] = y;
  if (T->lc[py] == y) T->lc[py] = x; else T->rc[py] = x;
  T->par[x] = py; T->par[y] = px;
}

/* apply NNI alternative j (0 or 1) at edge e.  Applying the same
   alternative a second time restores the original topology. */
static void lp_nni(LpTree *T, LpEdge *e, int j) {
  if (e->isroot)   /* v: children a,b; p: children c,d */
    lp_swap(T, T->rc[e->v], j == 0 ? T->lc[e->p] : T->rc[e->p]);
  else {           /* v: children a,b; sibling c */
    int p = e->p, c = (T->lc[p] == e->v ? T->rc[p] : T->lc[p]);
    lp_swap(T, j == 0 ? T->rc[e->v] : T->lc[e->v], c);
  }
}

/* build a TreeNode tree from the arrays and branch lengths, with node
   ids assigned as in the NJ code (leaves 0..n-1 in names order,
   internal nodes in postorder, root last).  If idmap is non-NULL it
   receives the id of the built node corresponding to each array node */
static TreeNode *lp_build_rec(LpTree *T, int v, double *bl, TreeNode **leaves,
                              int *idmap) {
  TreeNode *nd;
  if (T->lc[v] < 0)
    nd = leaves[T->leafidx[v]];
  else {
    TreeNode *a = lp_build_rec(T, T->lc[v], bl, leaves, idmap);
    TreeNode *b = lp_build_rec(T, T->rc[v], bl, leaves, idmap);
    nd = tr_new_node();
    tr_add_child(nd, a);
    tr_add_child(nd, b);
  }
  nd->dparent = (v == T->root ? 0 : bl[v]);
  if (idmap != NULL) idmap[v] = nd->id;
  return nd;
}

static TreeNode *lp_build(LpTree *T, double *bl, char **names, int *idmap) {
  int i;
  TreeNode **leaves = smalloc(T->n * sizeof(TreeNode*)), *root;
  tr_reset_id();
  for (i = 0; i < T->n; i++) {
    leaves[i] = tr_new_node();
    snprintf(leaves[i]->name, sizeof(leaves[i]->name), "%s", names[i]);
  }
  root = lp_build_rec(T, T->root, bl, leaves, idmap);
  root->nnodes = 2 * T->n - 1;
  tr_reset_nnodes(root);
  sfree(leaves);
  return root;
}

/* log likelihood of the tree defined by arrays and branch lengths;
   optionally fills grad (indexed by array node) with dlnL/dt */
static double lp_lnl(LpTree *T, double *bl, char **names, TreeModel *mod,
                     CovarData *data, double *grad) {
  double lnl;
  int *idmap = NULL, i;
  Vector *bg = NULL;
  TreeNode *t;

  if (grad != NULL) {
    idmap = smalloc(T->nn * sizeof(int));
    bg = vec_new(T->nn);
  }
  t = lp_build(T, bl, names, idmap);
  mod->tree = NULL;
  reset_tree_model(mod, t);
  if (data->crispr_mod != NULL) {
    assert(grad == NULL);
    lnl = cpr_compute_log_likelihood(data->crispr_mod, NULL);
  }
  else
    lnl = compute_log_likelihood(mod, data, bg);
  if (data->migtable != NULL &&
      !(data->crispr_mod != NULL && data->crispr_mod->mig_warmup))
    lnl += mig_compute_log_likelihood(mod, data->migtable, data->crispr_mod, NULL);
  if (grad != NULL) {
    for (i = 0; i < T->nn; i++)
      grad[i] = (i == T->root ? 0 : vec_get(bg, idmap[i]));
    vec_free(bg);
    sfree(idmap);
  }
  mod->tree = NULL;
  tr_free(t);
  return lnl;
}

/* finite-difference curvature h[i] = -d2 lnL / dt_i^2 for each free
   branch, given lnL l0 and gradient g at bl.  Uses one extra
   evaluation per branch: h = -2 (l(t+d) - l0 - g d) / d^2.  Returns
   the number of branches for which the curvature was not positive
   (these get a fallback value). */
static int lp_curvature(LpTree *T, double *bl, double l0, double *g,
                        int *isfree, char **names, TreeModel *mod,
                        CovarData *data, double *h) {
  int i, nfb = 0;
  double *bt = smalloc(T->nn * sizeof(double));
  memcpy(bt, bl, T->nn * sizeof(double));
  for (i = 0; i < T->nn; i++) {
    h[i] = 0;
    if (!isfree[i]) continue;
    double t = bl[i], d = fmax(LP_REL_STEP * t, LP_MIN_STEP), l1;
    bt[i] = t + d;
    l1 = lp_lnl(T, bt, names, mod, data, NULL);
    bt[i] = t;
    h[i] = -2.0 * (l1 - l0 - g[i] * d) / (d * d);
    if (!(h[i] > 0)) {  /* fallback: sd equal to max(t, d) */
      double s = fmax(t, d);
      h[i] = 1.0 / (s * s);
      nfb++;
    }
  }
  sfree(bt);
  return nfb;
}

/* ---------------------------------------------------------------------
   Ultrametric (rooted, clock-like) trees, used for CRISPR data and with
   --ultrametric.  The free parameters are the heights of the n-1
   internal nodes (leaves have height zero), so every sampled tree is
   ultrametric.  Gradients and curvatures are obtained by finite
   differences, the topological moves are rooted NNIs (exchanging a
   child of a non-root internal node with that node's sibling), and
   heights are sampled from the leaves upward, each truncated below by
   the heights of its children.  Because node heights are strongly
   correlated, each sampled tree is also rescaled by a global factor
   drawn from a Laplace approximation for the overall scale of the
   tree.  Branches at which no NNI configuration is favored (typically
   of zero length, e.g., where no mutations distinguish several cells)
   are merged into polytomies that are resolved at random in each
   sampled tree.
   --------------------------------------------------------------------- */

/* branch lengths from node heights */
static void lp_heights_to_bl(LpTree *T, double *H, double *bl) {
  for (int i = 0; i < T->nn; i++)
    bl[i] = (i == T->root ? 0 : fmax(0, H[T->par[i]] - H[i]));
}

static double lp_lnl_h(LpTree *T, double *H, double *bl, char **names,
                       TreeModel *mod, CovarData *data) {
  lp_heights_to_bl(T, H, bl);
  return lp_lnl(T, bl, names, mod, data, NULL);
}

static double lp_maxchild(LpTree *T, double *H, int v) {
  return fmax(H[T->lc[v]], H[T->rc[v]]);
}

/* postorder list of internal nodes */
static void lp_post_internal(LpTree *T, int v, int *out, int *k) {
  if (T->lc[v] < 0) return;
  lp_post_internal(T, T->lc[v], out, k);
  lp_post_internal(T, T->rc[v], out, k);
  out[(*k)++] = v;
}

/* finite-difference first and second derivatives of lnL with respect
   to the height of internal node v, respecting the ordering
   constraints.  l0 is lnL at H.  Returns FALSE if no usable step. */
static int lp_fd_height(LpTree *T, double *H, double *bl, int v, double l0,
                        double scale, char **names, TreeModel *mod,
                        CovarData *data, double *g, double *h) {
  double up = (v == T->root ? scale : H[T->par[v]] - H[v]),
    dn = H[v] - lp_maxchild(T, H, v), tiny = 1e-6 * scale, d, l1, l2,
    h0 = H[v];

  if (fmin(up, dn) > tiny) {          /* central */
    d = LP_REL_STEP * fmin(up, dn);
    H[v] = h0 + d; l1 = lp_lnl_h(T, H, bl, names, mod, data);
    H[v] = h0 - d; l2 = lp_lnl_h(T, H, bl, names, mod, data);
    *g = (l1 - l2) / (2 * d);
    *h = -(l1 - 2 * l0 + l2) / (d * d);
  }
  else if (up >= dn && up > tiny) {   /* forward */
    d = 0.5 * LP_REL_STEP * up;
    H[v] = h0 + d; l1 = lp_lnl_h(T, H, bl, names, mod, data);
    H[v] = h0 + 2 * d; l2 = lp_lnl_h(T, H, bl, names, mod, data);
    *g = (-3 * l0 + 4 * l1 - l2) / (2 * d);
    *h = -(l0 - 2 * l1 + l2) / (d * d);
  }
  else if (dn > tiny) {               /* backward */
    d = 0.5 * LP_REL_STEP * dn;
    H[v] = h0 - d; l1 = lp_lnl_h(T, H, bl, names, mod, data);
    H[v] = h0 - 2 * d; l2 = lp_lnl_h(T, H, bl, names, mod, data);
    *g = (3 * l0 - 4 * l1 + l2) / (2 * d);
    *h = -(l0 - 2 * l1 + l2) / (d * d);
  }
  else {
    H[v] = h0;
    return FALSE;
  }
  H[v] = h0;
  return TRUE;
}

/* derivatives for all internal nodes; nonpositive curvatures get a
   fallback.  Returns the number of fallbacks. */
static int lp_fd_all(LpTree *T, double *H, double *bl, double l0,
                     double scale, int *post, int npost, char **names,
                     TreeModel *mod, CovarData *data, double *g, double *h) {
  int k, nfb = 0;
  for (k = 0; k < npost; k++) {
    int v = post[k];
    double up = (v == T->root ? scale : H[T->par[v]] - H[v]),
      dn = H[v] - lp_maxchild(T, H, v), s;
    if (!lp_fd_height(T, H, bl, v, l0, scale, names, mod, data, &g[v], &h[v]))
      g[v] = h[v] = 0;
    if (!(h[v] > 0)) {
      s = fmax(fmax(fmin(up, dn), 0.5 * fmax(up, dn)), 1e-3 * scale);
      h[v] = 1.0 / (s * s);
      nfb++;
    }
  }
  return nfb;
}

/* replace the connected set of internal nodes mem[0..m-1] (with top node
   top) by a random binary resolution of the polytomy formed by their
   child subtrees, reusing the same node ids so that the rest of the
   tree is unchanged.  Topology is read from T0 and written to T.  The
   member nodes are marked as used. */
static void lp_resolve_polytomy(LpTree *T0, LpTree *T, int *mem, int m,
                                int top, int *comp, int c, int *used) {
  int *pool = smalloc((m + 1) * sizeof(int)), *ids = smalloc(m * sizeof(int)),
    np = 0, ni = 0, k;
  for (k = 0; k < m; k++) {
    int v = mem[k];
    used[v] = 1;
    if (comp[T0->lc[v]] != c) pool[np++] = T0->lc[v];
    if (comp[T0->rc[v]] != c) pool[np++] = T0->rc[v];
    if (v != top) ids[ni++] = v;
  }
  ids[ni++] = top;   /* the final join takes the top node's place */
  assert(np == m + 1 && ni == m);
  for (k = 0; k < m; k++) {   /* random sequential joins */
    int a = (int)(lp_unif() * np), x, y;
    x = pool[a]; pool[a] = pool[--np];
    a = (int)(lp_unif() * np);
    y = pool[a]; pool[a] = pool[--np];
    T->lc[ids[k]] = x; T->rc[ids[k]] = y;
    T->par[x] = ids[k]; T->par[y] = ids[k];
    pool[np++] = ids[k];
  }
  sfree(pool); sfree(ids);
}

static List *lp_sample_ultrametric(int nsamples, TreeNode *pointtree,
                                   TreeModel *mod, CovarData *data,
                                   char **names, unsigned int silent) {
  int n = data->nseqs, i, j, k, e, it, ne = 0, nfb, npost = 0;
  unsigned int subsample_saved = data->subsample;
  clock_t c0 = clock();
  LpTree *T0;
  double *H, *Hn, *bl, *g, *h, *P, lcur, lnew, l0, scale;
  int *post, *E;
  List *retval = lst_new_ptr(nsamples);

  data->subsample = FALSE;
  if (mod->tree != NULL) { tr_free(mod->tree); mod->tree = NULL; }

  T0 = lp_tree_new(pointtree, names, n);
  H = smalloc(T0->nn * sizeof(double)); Hn = smalloc(T0->nn * sizeof(double));
  bl = smalloc(T0->nn * sizeof(double));
  g = smalloc(T0->nn * sizeof(double)); h = smalloc(T0->nn * sizeof(double));
  post = smalloc(T0->nn * sizeof(int));

  /* heights from the point estimate */
  for (i = 0; i < T0->nn; i++) {
    TreeNode *nd = lst_get_ptr(pointtree->nodes, i);
    bl[nd->id] = (nd->parent != NULL ? fmax(nd->dparent, 0) : 0);
  }
  lp_post_internal(T0, T0->root, post, &npost);
  assert(npost == n - 1);
  for (i = 0; i < T0->nn; i++) H[i] = 0;
  for (k = 0; k < npost; k++) {
    int v = post[k];
    H[v] = fmax(H[T0->lc[v]] + bl[T0->lc[v]], H[T0->rc[v]] + bl[T0->rc[v]]);
  }
  scale = fmax(H[T0->root], 1e-6);
  memcpy(Hn, H, T0->nn * sizeof(double));   /* leaves stay at zero */

  /* 1. diagonal Newton iterations on the internal node heights */
  l0 = lcur = lp_lnl_h(T0, H, bl, names, mod, data);
  nfb = lp_fd_all(T0, H, bl, lcur, scale, post, npost, names, mod, data, g, h);
  for (it = 0; it < LP_MAX_NEWTON_H; it++) {
    double step = 1.0;
    for (;;) {
      for (k = 0; k < npost; k++) {  /* postorder keeps heights ordered */
        int v = post[k];
        Hn[v] = fmax(H[v] + step * g[v] / h[v], lp_maxchild(T0, Hn, v));
      }
      lnew = lp_lnl_h(T0, Hn, bl, names, mod, data);
      if (lnew >= lcur || step < 0.1) break;
      step *= 0.5;
    }
    if (lnew < lcur) break;   /* no improvement; g, h still current */
    for (k = 0; k < npost; k++) H[post[k]] = Hn[post[k]];
    double gain = lnew - lcur;
    lcur = lnew;
    /* 2. derivatives at the new heights (final ones define the Laplace
       approximation) */
    nfb = lp_fd_all(T0, H, bl, lcur, scale, post, npost, names, mod, data, g, h);
    if (gain < LP_NEWTON_TOL) break;
  }

  /* global scale of all node heights: the dominant correlated
     direction, which the per-node (conditional) variances miss */
  double sd_scale = 0;
  {
    double *Ht = smalloc(T0->nn * sizeof(double)), d = LP_SCALE_STEP, lp1, lm1, hs;
    for (i = 0; i < T0->nn; i++) Ht[i] = H[i] * exp(d);
    lp1 = lp_lnl_h(T0, Ht, bl, names, mod, data);
    for (i = 0; i < T0->nn; i++) Ht[i] = H[i] * exp(-d);
    lm1 = lp_lnl_h(T0, Ht, bl, names, mod, data);
    hs = -(lp1 - 2 * lcur + lm1) / (d * d);
    if (hs > 0) sd_scale = 1.0 / sqrt(hs);
    sfree(Ht);
  }

  /* 3. rooted NNI scores at each internal non-root node */
  E = smalloc(n * sizeof(int));
  for (k = 0; k < npost; k++)
    if (post[k] != T0->root) E[ne++] = post[k];
  assert(ne == n - 2);
  P = smalloc(3 * (ne > 0 ? ne : 1) * sizeof(double));
  int nunc = 0;
  for (e = 0; e < ne; e++) {
    int v = E[e], p = T0->par[v];
    double l[3], mx, s = 0, hv = H[v];
    LpEdge ed = {v, p, FALSE};
    l[0] = lcur;
    for (j = 0; j < 2; j++) {
      lp_nni(T0, &ed, j);
      H[v] = fmax(hv, lp_maxchild(T0, H, v));
      l[j+1] = lp_lnl_h(T0, H, bl, names, mod, data);
      H[v] = hv;
      lp_nni(T0, &ed, j);   /* undo */
    }
    mx = fmax(l[0], fmax(l[1], l[2]));
    for (j = 0; j < 3; j++) { P[3*e+j] = exp(l[j] - mx); s += P[3*e+j]; }
    for (j = 0; j < 3; j++) P[3*e+j] /= s;
    if (P[3*e] < 0.95) nunc++;
  }

  /* unresolved branches (no configuration favored, typically zero
     length) are merged into polytomies, which are resolved at random
     in each sampled tree */
  int *comp = smalloc(T0->nn * sizeof(int)), nclust = 0, npoly = 0;
  int **cmem = NULL, *csize = NULL, *ctop = NULL;
  for (i = 0; i < T0->nn; i++) comp[i] = i;
  for (e = 0; e < ne; e++) {
    if (P[3*e] >= LP_POLYTOMY_P) continue;
    int a = E[e], b = T0->par[E[e]];
    while (comp[a] != a) a = comp[a];
    while (comp[b] != b) b = comp[b];
    if (a != b) comp[a] = b;
    npoly++;
  }
  {
    int *cid = smalloc(T0->nn * sizeof(int)), *cnt = calloc(T0->nn, sizeof(int));
    for (i = 0; i < T0->nn; i++) {
      int r = i; while (comp[r] != r) r = comp[r];
      comp[i] = r; cnt[r]++;
    }
    for (i = 0; i < T0->nn; i++) cid[i] = -1;
    for (i = 0; i < T0->nn; i++)
      if (comp[i] == i && cnt[i] > 1) cid[i] = nclust++;
    cmem = smalloc((nclust > 0 ? nclust : 1) * sizeof(int*));
    csize = calloc(nclust > 0 ? nclust : 1, sizeof(int));
    ctop = smalloc((nclust > 0 ? nclust : 1) * sizeof(int));
    for (j = 0; j < nclust; j++) cmem[j] = smalloc(T0->nn * sizeof(int));
    for (i = 0; i < T0->nn; i++) {
      int c = cid[comp[i]];
      if (c < 0) { comp[i] = -1; continue; }
      cmem[c][csize[c]++] = i;
      comp[i] = c;
    }
    for (j = 0; j < nclust; j++)   /* top: member whose parent is outside */
      for (k = 0; k < csize[j]; k++) {
        int v = cmem[j][k];
        if (v == T0->root || comp[T0->par[v]] != j) ctop[j] = v;
      }
    sfree(cid); free(cnt);
  }

  if (!silent) {
    fprintf(stderr, "Local posterior (ultrametric): height optimization lnL %.2f -> %.2f (%d iterations)",
            l0, lcur, it);
    if (nfb > 0) fprintf(stderr, ", %d nodes with fallback curvature", nfb);
    fprintf(stderr, ";\n  global height-scale sd %.3f; %d of %d internal nodes with NNI support < 0.95,\n  %d unresolved, forming %d polytomies (%.1f s)\n",
            sd_scale, nunc, ne, npoly, nclust, (double)(clock() - c0) / CLOCKS_PER_SEC);
  }

  /* 4. sample trees */
  int *perm = smalloc((ne > 0 ? ne : 1) * sizeof(int)),
    *used = smalloc(T0->nn * sizeof(int)), *spost = smalloc(T0->nn * sizeof(int));
  double *Hs = smalloc(T0->nn * sizeof(double)), nswap = 0;
  for (int s = 0; s < nsamples; s++) {
    LpTree *T = lp_tree_copy(T0);
    int ns = 0;
    memset(used, 0, T0->nn * sizeof(int));
    for (j = 0; j < nclust; j++)
      lp_resolve_polytomy(T0, T, cmem[j], csize[j], ctop[j], comp, j, used);
    for (e = 0; e < ne; e++) perm[e] = e;
    for (e = ne - 1; e > 0; e--) {
      int r = (int)(lp_unif() * (e + 1)), tmp = perm[e];
      perm[e] = perm[r]; perm[r] = tmp;
    }
    for (int q = 0; q < ne; q++) {
      double u = lp_unif();
      int v;
      e = perm[q]; v = E[e];
      if (P[3*e] < LP_POLYTOMY_P) continue;   /* handled as a polytomy */
      j = (u < P[3*e] ? 0 : (u < P[3*e] + P[3*e+1] ? 1 : 2));
      if (j == 0 || used[v] || used[T->par[v]]) continue;
      LpEdge ed = {v, T->par[v], FALSE};
      lp_nni(T, &ed, j - 1);
      used[v] = used[ed.p] = 1;
      nswap++;
    }
    lp_post_internal(T, T->root, spost, &ns);
    /* heights from the leaves up, each truncated below by its children */
    for (i = 0; i < T0->nn; i++) Hs[i] = 0;
    for (k = 0; k < ns; k++) {
      int v = spost[k];
      double lb = lp_maxchild(T, Hs, v);
      Hs[v] = lb + lp_trunc_norm(H[v] + g[v] / h[v] - lb, 1.0 / sqrt(h[v]));
    }
    if (sd_scale > 0) {
      double f = exp(sd_scale * lp_norm());
      for (i = 0; i < T0->nn; i++) Hs[i] *= f;
    }
    lp_heights_to_bl(T, Hs, bl);
    lst_push_ptr(retval, lp_build(T, bl, names, NULL));
    lp_tree_free(T, FALSE);
  }

  if (!silent)
    fprintf(stderr, "Local posterior: %.1f NNI moves per sampled tree.\n",
            nswap / nsamples);

  data->subsample = subsample_saved;
  sfree(Hs); sfree(perm); sfree(used); sfree(spost); sfree(P); sfree(E);
  for (j = 0; j < nclust; j++) sfree(cmem[j]);
  sfree(cmem); free(csize); sfree(ctop); sfree(comp);
  sfree(H); sfree(Hn); sfree(bl); sfree(g); sfree(h); sfree(post);
  lp_tree_free(T0, TRUE);
  return retval;
}

List *lp_sample(int nsamples, TreeNode *pointtree, TreeModel *mod,
                CovarData *data, char **names, unsigned int silent) {
  int n = data->nseqs, i, j, e, it, ne = 0, nfb;
  unsigned int subsample_saved = data->subsample;
  clock_t c0 = clock();
  LpTree *T0;
  LpEdge *E;
  double *bl, *g, *h, *bnew, *P, lcur, lnew, l0;
  int *isfree;
  List *retval = lst_new_ptr(nsamples);

  if (n < 3)
    die("ERROR in lp_sample: need at least three taxa.\n");

  if (data->ultrametric)
    return lp_sample_ultrametric(nsamples, pointtree, mod, data, names, silent);

  data->subsample = FALSE;  /* exact likelihoods */
  if (mod->tree != NULL) { tr_free(mod->tree); mod->tree = NULL; }

  T0 = lp_tree_new(pointtree, names, n);
  bl = smalloc(T0->nn * sizeof(double));
  g = smalloc(T0->nn * sizeof(double));
  h = smalloc(T0->nn * sizeof(double));
  bnew = smalloc(T0->nn * sizeof(double));
  isfree = smalloc(T0->nn * sizeof(int));
  for (i = 0; i < T0->nn; i++) {
    TreeNode *nd = lst_get_ptr(pointtree->nodes, i);
    bl[nd->id] = (nd->parent != NULL ? fmax(nd->dparent, 0) : 0);
  }
  for (i = 0; i < T0->nn; i++)  /* all branches except root's right side */
    isfree[i] = (i != T0->root && i != T0->rc[T0->root]);

  /* 1. diagonal Newton iterations for the branch lengths */
  l0 = lcur = lp_lnl(T0, bl, names, mod, data, g);
  nfb = lp_curvature(T0, bl, lcur, g, isfree, names, mod, data, h);
  for (it = 0; it < LP_MAX_NEWTON; it++) {
    double step = 1.0;
    for (;;) {
      for (i = 0; i < T0->nn; i++)
        bnew[i] = isfree[i] ? fmax(0, bl[i] + step * g[i] / h[i]) : bl[i];
      lnew = lp_lnl(T0, bnew, names, mod, data, NULL);
      if (lnew >= lcur || step < 0.1) break;
      step *= 0.5;
    }
    if (lnew < lcur) break;   /* no improvement; keep current lengths */
    memcpy(bl, bnew, T0->nn * sizeof(double));
    if (lnew - lcur < LP_NEWTON_TOL) { lcur = lp_lnl(T0, bl, names, mod, data, g); break; }
    lcur = lp_lnl(T0, bl, names, mod, data, g);
  }

  /* 2. curvature at the optimized branch lengths */
  nfb = lp_curvature(T0, bl, lcur, g, isfree, names, mod, data, h);

  /* 3. NNI scores for each internal branch */
  E = smalloc(n * sizeof(LpEdge));
  for (i = 0; i < T0->nn; i++) {
    if (T0->lc[i] < 0 || i == T0->root || T0->par[i] == T0->root) continue;
    E[ne].v = i; E[ne].p = T0->par[i]; E[ne].isroot = FALSE; ne++;
  }
  if (n > 3 && T0->lc[T0->lc[T0->root]] >= 0 && T0->lc[T0->rc[T0->root]] >= 0) {
    E[ne].v = T0->lc[T0->root]; E[ne].p = T0->rc[T0->root];
    E[ne].isroot = TRUE; ne++;
  }
  assert(ne == n - 3);

  P = smalloc(3 * (ne > 0 ? ne : 1) * sizeof(double));
  int nunc = 0;
  for (e = 0; e < ne; e++) {
    double l[3], mx, s = 0;
    l[0] = lcur;
    for (j = 0; j < 2; j++) {
      lp_nni(T0, &E[e], j);
      l[j+1] = lp_lnl(T0, bl, names, mod, data, NULL);
      lp_nni(T0, &E[e], j);   /* undo */
    }
    mx = fmax(l[0], fmax(l[1], l[2]));
    for (j = 0; j < 3; j++) { P[3*e+j] = exp(l[j] - mx); s += P[3*e+j]; }
    for (j = 0; j < 3; j++) P[3*e+j] /= s;
    if (P[3*e] < 0.95) nunc++;
  }

  if (!silent) {
    fprintf(stderr, "Local posterior: branch-length optimization lnL %.2f -> %.2f (%d iterations)",
            l0, lcur, it);
    if (nfb > 0) fprintf(stderr, ", %d branches with fallback curvature", nfb);
    fprintf(stderr, ";\n  %d of %d internal branches with NNI support < 0.95 (%.1f s)\n",
            nunc, ne, (double)(clock() - c0) / CLOCKS_PER_SEC);
  }

  /* 4. sample trees */
  int *perm = smalloc((ne > 0 ? ne : 1) * sizeof(int)),
    *used = smalloc(T0->nn * sizeof(int));
  double *bs = smalloc(T0->nn * sizeof(double)), nswap = 0;
  for (int s = 0; s < nsamples; s++) {
    LpTree *T = lp_tree_copy(T0);
    memset(used, 0, T0->nn * sizeof(int));
    for (e = 0; e < ne; e++) perm[e] = e;
    for (e = ne - 1; e > 0; e--) {
      int r = (int)(lp_unif() * (e + 1)), tmp = perm[e];
      perm[e] = perm[r]; perm[r] = tmp;
    }
    for (int q = 0; q < ne; q++) {
      double u = lp_unif();
      e = perm[q];
      j = (u < P[3*e] ? 0 : (u < P[3*e] + P[3*e+1] ? 1 : 2));
      if (j == 0 || used[E[e].v] || used[E[e].p]) continue;
      lp_nni(T, &E[e], j - 1);
      used[E[e].v] = used[E[e].p] = 1;
      nswap++;
    }
    /* branch lengths travel with their branches through NNI moves */
    for (i = 0; i < T0->nn; i++)
      bs[i] = isfree[i] ? lp_trunc_norm(bl[i] + g[i] / h[i], 1.0 / sqrt(h[i])) : bl[i];
    lst_push_ptr(retval, lp_build(T, bs, names, NULL));
    lp_tree_free(T, FALSE);
  }

  if (!silent)
    fprintf(stderr, "Local posterior: %.1f NNI moves per sampled tree.\n",
            nswap / nsamples);

  data->subsample = subsample_saved;
  sfree(bs); sfree(perm); sfree(used); sfree(P); sfree(E);
  sfree(bl); sfree(g); sfree(h); sfree(bnew); sfree(isfree);
  lp_tree_free(T0, TRUE);
  return retval;
}
