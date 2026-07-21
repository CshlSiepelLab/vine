/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */


/* neighbor-joining implementations and supporting functions */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>
#include <crispr.h>
#include <likelihoods.h>
#include <backprop.h>
#include <upgma.h>

/* Record one neighbor-joining merge event into the Neighbors tape. */
static void nj_record_join(Neighbors *nb, int step_idx, int u, int v, int w,
                           int nactive, double d_uv, double row_sum_u,
                           double row_sum_v, int branch_idx_u,
                           int branch_idx_v) {
  if (step_idx < 0 || step_idx >= nb->nsteps)
    die("nj_record_join: step_idx (%d) out of range [0,%d)\n",
        step_idx, nb->nsteps);

  JoinEvent *ev = &nb->steps[step_idx];
  ev->u = u;
  ev->v = v;
  ev->w = w;
  ev->nk = nactive;
  ev->branch_idx_u = branch_idx_u;
  ev->branch_idx_v = branch_idx_v;

  assert(u < v);  /* enforced by logical cluster order */
  ev->d_uv = d_uv;
  ev->row_sum_u = row_sum_u;
  ev->row_sum_v = row_sum_v;
}

static void nj_find_min_q(Matrix *D, const int *active_ids, Vector *sums,
                          int nactive, int *u, int *v) {
  int a, b;
  double best = INFINITY;
  double scale = nactive - 2;
  double *sum = sums->data;

  *u = *v = -1;

  for (a = 0; a < nactive; a++) {
    int i = active_ids[a];
    double *row = D->data[i];
    double sum_i = sum[i];
    for (b = a+1; b < nactive; b++) {
      int j = active_ids[b];
      double q = scale * row[j] - sum_i - sum[j];
      if (q < best) {
        best = q;
        *u = i;
        *v = j;
      }
    }
  }

  if (*u < 0 || *v < 0)
    die("ERROR nj_find_min_q: fewer than two active taxa\n");
}

/* Infer a tree from a starting distance matrix. Reuse slot u for each newly
   joined cluster and retire slot v, so the working distance matrix remains
   n x n. Physical matrix slots are mapped to stable logical cluster IDs for
   recording and backpropagation. If inplace is false, does not alter the
   provided distance matrix. If dt_dD is non-NULL, populate it with the
   branch-length Jacobian. */
TreeNode *nj_infer(Matrix *initD, char **names, Matrix *dt_dD, Neighbors *nb,
                   unsigned int inplace) {
  int n = initD->nrows, nactive = n;
  int N = 2*n - 2, npairs = n * (n-1) / 2;
  int Npairs = N * (N-1) / 2;
  int i, j, a, slot_u = -1, slot_v = -1, w = n, step_idx = 0;
  Matrix *D;
  Vector *sums;
  Vector *logical_active = NULL;
  int *active_ids, *cluster_ids;
  TreeNode **nodes;
  TreeNode *root;
  static SparseMatrix *Jk = NULL, *Jnext = NULL;

  if (initD->nrows != initD->ncols || n < 3)
    die("ERROR nj_infer: bad distance matrix\n");
  if (dt_dD != NULL && (dt_dD->nrows != N || dt_dD->ncols != npairs))
    die("ERROR nj_infer: bad dimension in dt_dD\n");

  D = inplace ? initD : mat_new(n, n);
  sums = vec_new(n);
  active_ids = smalloc(n * sizeof(*active_ids));
  cluster_ids = smalloc(n * sizeof(*cluster_ids));
  nodes = smalloc(n * sizeof(*nodes));

  if (dt_dD != NULL) {
    logical_active = vec_new(N);
    vec_set_all(logical_active, FALSE);
    if (Jk != NULL && (Jk->nrows != Npairs || Jk->ncols != npairs)) {
      spmat_free(Jk);
      spmat_free(Jnext);
      Jk = Jnext = NULL;
    }
    if (Jk == NULL) {
      Jk = spmat_new(Npairs, npairs, 100);
      Jnext = spmat_new(Npairs, npairs, 100);
    }
    nj_backprop_init_sparse(Jk, n);
    mat_zero(dt_dD);
  }

  vec_zero(sums);
  tr_reset_id();
  for (i = 0; i < n; i++) {
    nodes[i] = tr_new_node();
    snprintf(nodes[i]->name, sizeof(nodes[i]->name), "%s", names[i]);
    active_ids[i] = i;
    cluster_ids[i] = i;
    if (logical_active != NULL)
      vec_set(logical_active, i, TRUE);
    mat_set(D, i, i, 0.0);
    for (j = i + 1; j < n; j++) {
      double d = mat_get(initD, i, j);
      mat_set(D, i, j, d);
      mat_set(D, j, i, d);
      sums->data[i] += d;
      sums->data[j] += d;
    }
  }

  while (nactive > 2) {
    TreeNode *node_u, *node_v, *node_w;
    double d_uv, limb_u, limb_v, sum_w = 0.0;
    int u, v;

    nj_find_min_q(D, active_ids, sums, nactive, &slot_u, &slot_v);
    u = cluster_ids[slot_u];
    v = cluster_ids[slot_v];
    d_uv = mat_get(D, slot_u, slot_v);
    /* Preserve the original arithmetic order for reproducibility. */
    limb_u = 0.5 * d_uv + 1.0 / (2.0 * (nactive - 2)) *
      (sums->data[slot_u] - sums->data[slot_v]);
    limb_v = d_uv - limb_u;
    if (signbit(limb_u)) limb_u = 0.0;
    if (signbit(limb_v)) limb_v = 0.0;

    node_u = nodes[slot_u];
    node_v = nodes[slot_v];
    node_w = tr_new_node();
    tr_add_child(node_w, node_u);
    tr_add_child(node_w, node_v);
    node_u->dparent = limb_u;
    node_v->dparent = limb_v;

    if (nb != NULL) {
      nj_record_join(nb, step_idx, u, v, w, nactive, d_uv,
                     sums->data[slot_u], sums->data[slot_v],
                     node_u->id, node_v->id);
      step_idx++;
    }
    if (dt_dD != NULL) {
      nj_backprop_set_dt_dD_sparse(Jk, dt_dD, n, u, v,
                                   node_u->id, node_v->id, logical_active);
      nj_backprop_sparse(Jk, Jnext, n, u, v, w, logical_active);
    }

    for (a = 0; a < nactive; a++) {
      int k = active_ids[a];
      double du, dv, dw;
      if (k == slot_u || k == slot_v) continue;
      du = mat_get(D, slot_u, k);
      dv = mat_get(D, slot_v, k);
      dw = 0.5 * (du + dv - d_uv);
      if (dw < 0.0) dw = 0.0;
      mat_set(D, slot_u, k, dw);
      mat_set(D, k, slot_u, dw);
      sums->data[k] += dw - du - dv;
      sum_w += dw;
    }
    sums->data[slot_u] = sum_w;
    nodes[slot_u] = node_w;
    nodes[slot_v] = NULL;
    cluster_ids[slot_u] = w;

    if (logical_active != NULL) {
      vec_set(logical_active, u, FALSE);
      vec_set(logical_active, v, FALSE);
      vec_set(logical_active, w, TRUE);
    }
    /* Remove both old logical clusters, then append the new cluster's slot. */
    j = 0;
    for (a = 0; a < nactive; a++)
      if (active_ids[a] != slot_u && active_ids[a] != slot_v)
        active_ids[j++] = active_ids[a];
    active_ids[j] = slot_u;
    nactive--;
    w++;

    if (dt_dD != NULL) {
      SparseMatrix *tmp = Jk;
      Jk = Jnext;
      Jnext = tmp;
    }
  }

  slot_u = active_ids[0];
  slot_v = active_ids[1];
  root = tr_new_node();
  tr_add_child(root, nodes[slot_u]);
  tr_add_child(root, nodes[slot_v]);
  nodes[slot_u]->dparent = mat_get(D, slot_u, slot_v) / 2.0;
  nodes[slot_v]->dparent = mat_get(D, slot_u, slot_v) / 2.0;
  if (nb != NULL) {
    nb->nsteps = step_idx;
    nb->root_u = cluster_ids[slot_u];
    nb->root_v = cluster_ids[slot_v];
    nb->branch_idx_root_u = nodes[slot_u]->id;
    nb->branch_idx_root_v = nodes[slot_v]->id;
  }
  if (dt_dD != NULL)
    nj_backprop_set_dt_dD_sparse(Jk, dt_dD, n,
                                 cluster_ids[slot_u], cluster_ids[slot_v],
                                 nodes[slot_u]->id, nodes[slot_v]->id,
                                 logical_active);
  root->nnodes = 2*n - 1;
  tr_reset_nnodes(root);
  assert(root->id == root->nnodes - 1);

  sfree(nodes);
  sfree(cluster_ids);
  sfree(active_ids);
  if (logical_active != NULL) vec_free(logical_active);
  vec_free(sums);
  if (!inplace) mat_free(D);
  return root;
}

/* compute pairwise distance between two DNA seqs using the
   Jukes-Cantor model */
double compute_JC_dist(MSA *msa, int i, int j) {
  int k, diff = 0, n = 0;
  double d;
  for (k = 0; k < msa->length; k++) {
    if (msa->seqs[i][k] == GAP_CHAR || msa->seqs[j][k] == GAP_CHAR ||
        msa->is_missing[(int)msa->seqs[i][k]] ||
        msa->is_missing[(int)msa->seqs[j][k]])
      continue;
    n++;
    if (msa->seqs[i][k] != msa->seqs[j][k])
      diff++;
  }
  if (n == 0 || (double)diff/n >= 0.75)
    /* either no comparable sites (all gaps/missing for this pair) or
       too many differences for the JC correction to work.  We want
       the distance to be "really long" but not so long that it
       completely skews the tree or makes it impossible for the
       variational algorithm to recover. */
    d = 3.0;
  else
    d = -0.75 * log(1 - 4.0/3 * diff/n);   /* Jukes-Cantor correction */

  assert(isfinite(d));
  return d;
}

/* based on a multiple alignment, build and return a distance matrix
   using the Jukes-Cantor model.  Assume DNA alphabet */
Matrix *compute_JC_matr(MSA *msa) {
  int i, j;
  Matrix *retval = mat_new(msa->nseqs, msa->nseqs);

  mat_zero(retval);
  
  for (i = 0; i < msa->nseqs; i++) 
    for (j = i+1; j < msa->nseqs; j++) 
      mat_set(retval, i, j,
              compute_JC_dist(msa, i, j));

  return retval;  
}

/* compute a distance matrix from a tree, defining each pairwise
   distance as the edge length between the corresponding taxa */ 
Matrix *tree_to_distances(TreeNode *tree, char **names, int n) {
  TreeNode *n1, *n2;
  List *leaves = lst_new_ptr(tree->nnodes);
  int i, j, ii, jj;
  Matrix *D;
  double dist;
  int *seq_idx;
  unsigned int all_zeroes = TRUE;
  
  assert(tree->nodes != NULL);  /* assume list of nodes exists */
  
  for (i = 0; i < tree->nnodes; i++) {
    n1 = lst_get_ptr(tree->nodes, i);
    if (all_zeroes == TRUE && n1->dparent > 0) all_zeroes = FALSE;
    if (n1->lchild == NULL && n1->rchild == NULL)
      lst_push_ptr(leaves, n1);
  }
  
  if (lst_size(leaves) != n)
    die("ERROR in tree_to_distances: number of names must match number of leaves in tree.\n");

  /* if the input tree had no branch lengths defined, all will have
     values of zero, which will be a problem.  In this case,
     just initialize them all to a small constant value */
  if (all_zeroes == TRUE) {
    for (i = 0; i < tree->nnodes; i++) {
      n1 = lst_get_ptr(tree->nodes, i);
      if (n1->parent != NULL)
        n1->dparent = 0.1;
    }
  }
  
  D = mat_new(n, n);
  mat_zero(D);

  seq_idx = build_seq_idx(leaves, names);

  /* O(n^2) operation but seems plenty fast in practice */
  for (i = 0; i < lst_size(leaves); i++) {
    n1 = lst_get_ptr(leaves, i);
    ii = seq_idx[n1->id];   /* convert to seq indices for matrix */
    for (j = i+1; j < lst_size(leaves); j++) {
      n2 = lst_get_ptr(leaves, j);
      jj = seq_idx[n2->id];
      dist = distance_on_tree(tree, n1, n2);
      if (ii < jj)
        mat_set(D, ii, jj, dist);
      else
        mat_set(D, jj, ii, dist);
    }
  }

  lst_free(leaves);
  sfree(seq_idx);
  
  return(D);
}

double distance_on_tree(TreeNode *root, TreeNode *n1, TreeNode *n2) {
  double dist[root->nnodes];
  int id;
  TreeNode *n;
  double totd1, totd2;
  
  /* initialize distance from n1 to each ancestor to be -1 */
  for (id = 0; id < root->nnodes; id++)
    dist[id] = -1;
  
  /* find distance to each ancestor of n1 */
  for (n = n1, totd1 = 0; n->parent != NULL; n = n->parent) {
    totd1 += n->dparent;
    dist[n->parent->id] = totd1;
  }
  dist[root->id] = totd1;

  /* now trace ancestry of n2 until an ancestor of n1 is found */
  for (n = n2, totd2 = 0; dist[n->id] == -1 && n->parent != NULL; n = n->parent) 
    totd2 += n->dparent;

  if (n->parent == NULL && dist[n->id] == -1)
    die("ERROR in distance_on_tree: got to root without finding LCA\n");
  
  /* at this point, it must be true that n is the LCA of n1 and n2 */ 
  return totd2 + dist[n->id];
  
}

/* wrapper for various distance-based tree inference algorithms */
TreeNode *infer_distance_tree(Matrix *D, char **names, Matrix *dt_dD, Neighbors *nb,
                 CovarData *data) {
  if (data->ultrametric) {
    TreeNode *t = upgma_infer(D, names, dt_dD);

    if (data->no_zero_br == TRUE)
      repair_zero_br(t);
    return t;
  }
  else {
    TreeNode *tree = nj_infer(D, names, dt_dD, nb, FALSE);
    if (data->treeprior != NULL && data->treeprior->relclock == TRUE) { /* need to reroot in this case */
      if (data->seq_to_node_map == NULL) /* only need to do this once */
        update_seq_to_node_map(tree, names, data);
      if (data->tree_diam_leaf1 < 0 || data->tree_diam_leaf2 < 0)
        update_diam_leaves(D, data);  /* needed upon init */
      TreeNode *mp = tr_find_midpoint(tree, data->seq_to_node_map[data->tree_diam_leaf1],
                                      data->seq_to_node_map[data->tree_diam_leaf2]);
      TreeNode *newtree = tr_reroot2(tree, mp);
      tree = newtree;
    }
    return tree;
  }
}

void update_seq_to_node_map(TreeNode *tree, char **names, CovarData *data) {
  List *leaves = lst_new_ptr(tree->nnodes/2);
  if (data->seq_to_node_map != NULL)
    sfree(data->seq_to_node_map);

  /* collect leaves */
  for (int i = 0; i < tree->nnodes; i++) {
    TreeNode *n = lst_get_ptr(tree->nodes, i);
    if (n->lchild == NULL && n->rchild == NULL)
      lst_push_ptr(leaves, n);
  }

  /* obtain mapping from leaf node ids to seq idxs */
  int *seq_idx = build_seq_idx(leaves, names);
  
  /* we need the inverse of this mapping */
  data->seq_to_node_map = smalloc(data->msa->nseqs * sizeof(int));
  for (int i = 0; i < lst_size(leaves); i++) {
    TreeNode *n = lst_get_ptr(leaves, i);
    int seqidx = seq_idx[n->id];
    data->seq_to_node_map[seqidx] = n->id;
  }

  lst_free(leaves);
  sfree(seq_idx);
}

/* find leaves corresponding to largest distance in matrix.  Generally
   this will be done as a side effect of points_to_distances but upon initialization
   it needs to be done separately */
void update_diam_leaves(Matrix *D, CovarData *data) {
  double maxdist = 0;
  for (int i = 0; i < D->nrows; i++) {
    for (int j = i+1; j < D->ncols; j++) {
      double dist = mat_get(D, i, j);
      if (dist > maxdist) {
        maxdist = dist;
        data->tree_diam_leaf1 = i; 
        data->tree_diam_leaf2 = j;
      }
    }
  }
}

void repair_zero_br(TreeNode *t) {
  for (int nodeidx = 0; nodeidx < lst_size(t->nodes); nodeidx++) {
    TreeNode *n = lst_get_ptr(t->nodes, nodeidx);
    if (n->parent != NULL && n->dparent < 0.0)
      n->dparent = 0.0;
  }
}

/* functions to record neighbor information to facilitate backpropagation */

/* Allocate and initialize a Neighbors recorder for NJ on n taxa. */
Neighbors *nj_new_neighbors(int n) {
  Neighbors *nb = (Neighbors *)smalloc(sizeof(Neighbors));
  nb->n           = n;
  nb->total_nodes = 2*n - 2;
  nb->nsteps = n-2;  /* This is the max; will be reset after NJ run */

  nb->steps = (JoinEvent *)smalloc((n-2) * sizeof(JoinEvent));

  nb->root_u = nb->root_v = -1;
  nb->branch_idx_root_u = nb->branch_idx_root_v = -1;
  
  return nb;
}

/* Optional helper to free a Neighbors recorder when you’re done. */
void nj_free_neighbors(Neighbors *nb) {
  if (nb == NULL) return;
  free(nb->steps);
  free(nb);
}

void nj_copy_neighbors(Neighbors *dest, Neighbors *src) {
  dest->n = src->n;
  dest->total_nodes = src->total_nodes;
  dest->nsteps = src->nsteps;
  dest->root_u = src->root_u;
  dest->root_v = src->root_v;
  dest->branch_idx_root_u = src->branch_idx_root_u;
  dest->branch_idx_root_v = src->branch_idx_root_v;

  memcpy(dest->steps, src->steps, src->nsteps * sizeof(JoinEvent));
}
