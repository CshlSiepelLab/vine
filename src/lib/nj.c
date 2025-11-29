/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
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
#include <likelihoods.h>
#include <backprop.h>
#include <heap.h>
#include <upgma.h>

/* Reset Q matrix based on distance matrix.  Assume upper triangular
   square Q and D.  Only touches active rows and columns of Q and D up
   to maxidx. As a side-effect set u and v to the indices of the
   closest neighbors.  Also update sums to sum of distances from each
   node */
void nj_resetQ(Matrix *Q, Matrix *D, Vector *active, Vector *sums, int *u,
               int *v, int maxidx) {
  int i, j, n = 0;
  double min = INFINITY;
  
  if ((D->nrows != D->ncols) || (D->nrows != Q->nrows) ||
      (D->nrows != Q->ncols))
    die("ERROR nj_setQ: dimension mismatch\n");

  *u = *v = 0;
  
  /* update row sums */
  vec_zero(sums);
  for (i = 0; i < maxidx; i++) {
    if (vec_get(active, i) == TRUE) {
      n++;
      for (j = i+1; j < maxidx; j++) {
        if (vec_get(active, j) == TRUE) {
          sums->data[i] += mat_get(D, i, j);
          sums->data[j] += mat_get(D, i, j);
        }
      }
    }
  }
  
  /* now reset Q */
  for (i = 0; i < maxidx; i++) {
    if (vec_get(active, i) == TRUE) {
      for (j = i+1; j < maxidx; j++) {
        if (vec_get(active, j) == TRUE) {
          double qij = (n-2) * mat_get(D, i, j) - vec_get(sums, i) -
            vec_get(sums, j);
          mat_set(Q, i, j, qij);
          if (qij < min) {
            min = qij;
            *u = i;
            *v = j;
          }
        }
      }
    }
  }

  if (min == INFINITY)
    die("ERROR in nj_resetQ: fewer than two active taxa\n");
}

/* Update distance matrix after operation that joins neighbors u and v
   and adds new node w.  Update active list accordingly. Assumes u < v
   < w.  Also assumes all nodes > w are inactive. Assumes sums are
   precomputed */
void nj_updateD(Matrix *D, int u, int v, int w, Vector *active, Vector *sums) {
  int k;
  int n = vec_sum(active);

  if (D->nrows != D->ncols)
    die("ERROR nj_updateD: dimension mismatch\n");
  if (v <= u || w <= v)
    die("ERROR nj_updateD: indices out of order\n");
  if (n <= 2)
    die("ERROR nj_updateD: too few active nodes\n");
  
  mat_set(D, u, w, 0.5 * mat_get(D, u, v) +
	  1.0/(2.0*(n-2)) * (vec_get(sums, u) - vec_get(sums, v)));

  mat_set(D, v, w, mat_get(D, u, v) - mat_get(D, u, w));

  /* we can't let the distances go negative in this implementation
     because it will mess up the likelihood calculation */
  if (signbit(mat_get(D, u, w))) /* covers -0 case */
    mat_set(D, u, w, 0);
  if (signbit(mat_get(D, v, w)))
    mat_set(D, v, w, 0);
  
  for (k = 0; k < w; k++) {
    if (vec_get(active, k) == TRUE && k != u && k != v) {
      double du, dv;

      /* needed because of upper triangular constraint */
      du = (u < k ? mat_get(D, u, k) : mat_get(D, k, u));
      dv = (v < k ? mat_get(D, v, k) : mat_get(D, k, v));
      
      mat_set(D, k, w, 0.5 * (du + dv - mat_get(D, u, v)));

      if (mat_get(D, k, w) < 0)
        mat_set(D, k, w, 0);
    }
  }
}


/* Main function to infer the tree from a starting distance matrix.
   Does not alter the provided distance matrix.  If dt_dD is non-NULL,
   will be populated with Jacobian for 2n-3 branch lengths
   vs. n-choose-2 pairwise distances */
TreeNode* nj_infer_tree(Matrix *initD, char **names, Matrix *dt_dD) {
    int n = initD->nrows;
    int N = 2*n - 2;   /* number of nodes in unrooted tree */
    int i, j, u = -1, v = -1, w;
    Matrix *D, *Q;
    Vector *sums, *active;
    List *nodes;  /* could just be an array */
    TreeNode *node_u, *node_v, *node_w, *root;
    int npairs = n * (n-1) / 2, Npairs = N * (N-1) / 2;
    double *Jk = NULL, *Jnext = NULL;
    
    if (initD->nrows != initD->ncols || n < 3)
      die("ERROR nj_infer_tree: bad distance matrix\n");

    if (dt_dD != NULL && (dt_dD->nrows != N || dt_dD->ncols != npairs))
      die("ERROR nj_infer_tree: bad dimension in dt_dD\n");
    
    /* create a larger distance matrix of dimension N x N to
       accommodate internal nodes; also set up list of active nodes
       and starting tree nodes */
    D = mat_new(N, N); mat_zero(D);
    active = vec_new(N); vec_set_all(active, FALSE);
    sums = vec_new(N); vec_zero(sums);
    nodes = lst_new_ptr(N);
    tr_reset_id();
    
    for (i = 0; i < n; i++) {
      node_u = tr_new_node();
      strcat(node_u->name, names[i]);
      lst_push_ptr(nodes, node_u);
      vec_set(active, i, TRUE);
      for (j = i+1; j < n; j++)
        mat_set(D, i, j, mat_get(initD, i, j));
    }
   
    /* set up Q */
    Q = mat_new(N, N); mat_zero(Q);

    /* set up backprop data */
    if (dt_dD != NULL) {
      Jk = malloc(Npairs * npairs * sizeof(double));
      Jnext = malloc(Npairs * npairs * sizeof(double));
      nj_backprop_init(Jk, n);
      mat_zero(dt_dD);
    }
    
    /* main loop, over internal nodes w */
    for (w = n; w < N; w++) {   
      nj_resetQ(Q, D, active, sums, &u, &v, w);
      
      nj_updateD(D, u, v, w, active, sums);                    
      node_w = tr_new_node();
      lst_push_ptr(nodes, node_w);

      /* attach child nodes to parent and set branch lengths */
      node_u = lst_get_ptr(nodes, u);
      node_v = lst_get_ptr(nodes, v);
      tr_add_child(node_w, node_u);
      tr_add_child(node_w, node_v);
      node_u->dparent = mat_get(D, u, w);
      node_v->dparent = mat_get(D, v, w);

      if (dt_dD != NULL) {
        nj_backprop_set_dt_dD(Jk, dt_dD, n, u, v, node_u->id, node_v->id, active);
        nj_backprop(Jk, Jnext, n, u, v, w, active);
      }

      /* this has to be done after the backprop calls */
      vec_set(active, u, FALSE);
      vec_set(active, v, FALSE);
      vec_set(active, w, TRUE);

      if (dt_dD != NULL) {
        free(Jk);
        Jk = Jnext;
        Jnext = malloc(Npairs * npairs * sizeof(double));
      }
    }

    /* there should be exactly two active nodes left. Join them under
       a root. */
    node_u = NULL; node_v = NULL;
    root = tr_new_node();
    for (i = 0; i < N; i++) {
      if (vec_get(active, i) == TRUE) {
        if (node_u == NULL) {
          u = i;
          node_u = lst_get_ptr(nodes, i);
        }
        else if (node_v == NULL) {
          v = i;
          node_v = lst_get_ptr(nodes, i);
        }
        else
          die("ERROR nj_infer_tree: more than two nodes left at root\n");
      }
    }
    tr_add_child(root, node_u);
    tr_add_child(root, node_v);
    node_u->dparent = mat_get(D, u, v) / 2;
    node_v->dparent = mat_get(D, u, v) / 2;

    if (dt_dD != NULL) 
      nj_backprop_set_dt_dD(Jk, dt_dD, n, u, v, node_u->id, node_v->id, active);
    
    /* finish set up of tree */
    root->nnodes = N+1;
    tr_reset_nnodes(root);

    assert(root->id == root->nnodes - 1); /* important for indexing */
    
    lst_free(nodes);
    vec_free(active);
    vec_free(sums);
    mat_free(D);
    mat_free(Q);

    if (dt_dD != NULL) {
      free(Jk);
      free(Jnext);
    }
    
    return root;
}

/* Faster version of function to infer the tree from a starting
   distance matrix. Uses a min-heap for efficient lookup of minimum Q
   values, with lazy evaluation to avoid unnecessary computations.
   Does not alter the provided distance matrix. If dt_dD is non-NULL,
   will be populated with Jacobian for 2n-3 branch lengths
   vs. n-choose-2 pairwise distances */
TreeNode* nj_fast_infer(Matrix *initD, char **names, Matrix *dt_dD) {
  int n = initD->nrows, orign = n;
  int N = 2*n - 2;   /* number of nodes in unrooted tree */
  int i, j, u = -1, v = -1, w;
  Matrix *D;
  Vector *sums, *active;
  List *nodes;  
  TreeNode *node_u, *node_v, *node_w, *root;
  HeapNode *heap = NULL;
  NJHeapNode *hn, *newhn;
  int rev[N];
  int npairs = n * (n-1) / 2, Npairs = N * (N-1) / 2;
  //  double *Jk = NULL, *Jnext = NULL;
  static SparseMatrix *Jk = NULL, *Jnext = NULL;
    
  if (initD->nrows != initD->ncols || n < 3)
    die("ERROR nj_fast_infer: bad distance matrix\n");

  if (dt_dD != NULL && (dt_dD->nrows != N || dt_dD->ncols != npairs))
    die("ERROR nj_fast_infer: bad dimension in dt_dD\n");
    
  /* initialize revision numbers for all nodes */
  for (i = 0; i < N; i++) rev[i] = 0;

  /* create a larger distance matrix of dimension N x N to
     accommodate internal nodes; also set up list of active nodes
     and starting tree nodes */
  D = mat_new(N, N); mat_zero(D);
  active = vec_new(N); vec_set_all(active, FALSE);
  sums = vec_new(N); vec_zero(sums);
  nodes = lst_new_ptr(N);
  tr_reset_id();
    
  for (i = 0; i < n; i++) {
    node_u = tr_new_node();
    strcat(node_u->name, names[i]);
    lst_push_ptr(nodes, node_u);
    vec_set(active, i, TRUE);
    for (j = i+1; j < n; j++) {
      double d = mat_get(initD, i, j);
      mat_set(D, i, j, d);
      sums->data[i] += d;
      sums->data[j] += d;
    }
  }

  /* also set up the heap for Q values */
  for (i = 0; i < n; i++) {
    for (j = i+1; j < n; j++) {
      hn = nj_heap_computeQ(i, j, n, D, sums, rev);
      heap = hp_insert(heap, hn->val, hn);
    }
  }

  /* set up backprop data */
  if (dt_dD != NULL) {
    if (Jk == NULL) { /* first call */
      Jk = spmat_new(Npairs, npairs, 100);
      Jnext = spmat_new(Npairs, npairs, 100);
    }
    else
      assert(Npairs == Jk->nrows && npairs == Jk->ncols);

    nj_backprop_init_sparse(Jk, n);
    mat_zero(dt_dD);
  }
    
  /* main loop, over internal nodes w */
  for (w = n; w < N; w++) {

    /* get the minimum Q value from the heap; use lazy evaluation */
    while (TRUE) {
      heap = hp_delete_min(heap, (void**)&hn);

      if (vec_get(active, hn->i) == FALSE || vec_get(active, hn->j) == FALSE) {
        free(hn);
        continue;
      }
      else if (hn->rev_i == rev[hn->i] && hn->rev_j == rev[hn->j]) 
        break; /* valid and active */
      else {
        /* active but stale; recompute */
        newhn = nj_heap_computeQ(hn->i, hn->j, n, D, sums, rev);
        heap = hp_insert(heap, newhn->val, newhn);
        free(hn);
      }
    }
      
    /* join u and v; w is the new node */
    u = hn->i;
    v = hn->j;
    nj_updateD(D, u, v, w, active, sums);
    node_w = tr_new_node();
    lst_push_ptr(nodes, node_w);

    /* attach child nodes to parent and set branch lengths */
    node_u = lst_get_ptr(nodes, u);
    node_v = lst_get_ptr(nodes, v);
    tr_add_child(node_w, node_u);
    tr_add_child(node_w, node_v);
    node_u->dparent = mat_get(D, u, w);
    node_v->dparent = mat_get(D, v, w);
    
    /* update row sums and revision numbers */
    vec_set(sums, w, 0);  
    for (i = 0; i < w; i++) {
      if (vec_get(active, i) == TRUE && i != u && i != v) {   
        double du = (u < i ? mat_get(D, u, i) : mat_get(D, i, u)); /* upper triangular */
        double dv = (v < i ? mat_get(D, v, i) : mat_get(D, i, v));
        sums->data[i] += (mat_get(D, i, w) - du - dv); /* can be updated */
        sums->data[w] += mat_get(D, i, w); /* have to compute from scratch */
      }
      rev[i]++;
    }
    rev[w]++;
    
    if (dt_dD != NULL) {
            nj_backprop_set_dt_dD_sparse(Jk, dt_dD, orign, u, v, node_u->id, node_v->id, active);
            nj_backprop_sparse(Jk, Jnext, orign, hn->i, hn->j, w, active);
    }

    /* this has to be done after the backprop calls */
    vec_set(active, u, FALSE);
    vec_set(active, v, FALSE);
    vec_set(active, w, TRUE);
    n--;  /* one fewer active nodes */

    if (dt_dD != NULL) {
      /* swap pointers to avoid deep copy */
      SparseMatrix *tmp = Jk; 
      Jk = Jnext; 
      Jnext = tmp;
    }
      
    /* finally, add new Q values to the heap */
    for (i = 0; i < w; i++) {
      if (vec_get(active, i) == TRUE) {
        newhn = nj_heap_computeQ(i, w, n, D, sums, rev); /* add to heap */
        heap = hp_insert(heap, newhn->val, newhn);
      }
    }

    free(hn);
  }

  /* there should be exactly two active nodes left. Join them under
     a root. */
  node_u = NULL; node_v = NULL;
  root = tr_new_node();
  for (i = 0; i < N; i++) {
    if (vec_get(active, i) == TRUE) {
      if (node_u == NULL) {
        u = i;
        node_u = lst_get_ptr(nodes, i);
      }
      else if (node_v == NULL) {
        v = i;
        node_v = lst_get_ptr(nodes, i);
      }
      else
        die("ERROR nj_fast_infer: more than two nodes left at root\n");
    }
  }
  tr_add_child(root, node_u);
  tr_add_child(root, node_v);
  node_u->dparent = mat_get(D, u, v) / 2;
  node_v->dparent = mat_get(D, u, v) / 2;

  if (dt_dD != NULL) 
    nj_backprop_set_dt_dD_sparse(Jk, dt_dD, orign, u, v, node_u->id, node_v->id, active);
  //nj_backprop_set_dt_dD(Jk, dt_dD, orign, u, v, node_u->id, node_v->id, active);
  
  /* finish set up of tree */
  root->nnodes = N+1;
  tr_reset_nnodes(root);

  assert(root->id == root->nnodes - 1); /* important for indexing */

  /* drain heap */
  while (heap != NULL) {
    heap = hp_delete_min(heap, (void**)&hn);
    free(hn);
  }
  
  hp_free(heap);
  lst_free(nodes);
  vec_free(active);
  vec_free(sums);
  mat_free(D);
    
  return root;
}

NJHeapNode* nj_heap_computeQ(int i, int j, int n, Matrix *D, Vector *sums, int *rev) {
    NJHeapNode *hn = malloc(sizeof(NJHeapNode));
    hn->i = i;
    hn->j = j;
    hn->val = (n - 2) * mat_get(D, i, j) - vec_get(sums, i) -
      vec_get(sums, j);
    hn->rev_i = rev[i];
    hn->rev_j = rev[j];
    return hn;
}

/* compute pairwise distance between two DNA seqs using the
   Jukes-Cantor model */
double nj_compute_JC_dist(MSA *msa, int i, int j) {
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
  if ((double)diff/n >= 0.75)
    /* in this case, there are too many differences for the correction
       to work.  We basically want the distance to be "really long" but
       not so long that it completely skews the tree or makes it
       impossible for the variational algorithm to recover.  */
    d = 3.0;
  else
    d = -0.75 * log(1 - 4.0/3 * diff/n);   /* Jukes-Cantor correction */

  assert(isfinite(d));
  return d;
}

/* based on a multiple alignment, build and return a distance matrix
   using the Jukes-Cantor model.  Assume DNA alphabet */
Matrix *nj_compute_JC_matr(MSA *msa) {
  int i, j;
  Matrix *retval = mat_new(msa->nseqs, msa->nseqs);

  mat_zero(retval);
  
  for (i = 0; i < msa->nseqs; i++) 
    for (j = i+1; j < msa->nseqs; j++) 
      mat_set(retval, i, j,
              nj_compute_JC_dist(msa, i, j));

  return retval;  
}

/* compute a distance matrix from a tree, defining each pairwise
   distance as the edge length between the corresponding taxa */ 
Matrix *nj_tree_to_distances(TreeNode *tree, char **names, int n) {
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
    die("ERROR in nj_tree_to_distances: number of names must match number of leaves in tree.\n");

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

  seq_idx = nj_build_seq_idx(leaves, names);

  /* O(n^2) operation but seems plenty fast in practice */
  for (i = 0; i < lst_size(leaves); i++) {
    n1 = lst_get_ptr(leaves, i);
    ii = seq_idx[n1->id];   /* convert to seq indices for matrix */
    for (j = i+1; j < lst_size(leaves); j++) {
      n2 = lst_get_ptr(leaves, j);
      jj = seq_idx[n2->id];
      dist = nj_distance_on_tree(tree, n1, n2);
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

double nj_distance_on_tree(TreeNode *root, TreeNode *n1, TreeNode *n2) {
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
    die("ERROR in nj_distance_on_tree: got to root without finding LCA\n");
  
  /* at this point, it must be true that n is the LCA of n1 and n2 */ 
  return totd2 + dist[n->id];
  
}

/* wrapper for various distance-based tree inference algorithms */
TreeNode *nj_inf(Matrix *D, char **names, Matrix *dt_dD,
                 CovarData *data) {
  if (data->ultrametric) {
    TreeNode *t = upgma_fast_infer(D, names, dt_dD);

    if (data->no_zero_br == TRUE)
      nj_repair_zero_br(t);
    return t;
  }
  else {
    TreeNode *tree = nj_fast_infer(D, names, dt_dD);
    if (data->treeprior != NULL && data->treeprior->relclock == TRUE) { /* need to reroot in this case */
      if (data->seq_to_node_map == NULL) /* only need to do this once */
        nj_update_seq_to_node_map(tree, names, data);
      if (data->tree_diam_leaf1 < 0 || data->tree_diam_leaf2 < 0)
        nj_update_diam_leaves(D, data);  /* needed upon init */
      TreeNode *mp = tr_find_midpoint(tree, data->seq_to_node_map[data->tree_diam_leaf1],
                                      data->seq_to_node_map[data->tree_diam_leaf2]);
      TreeNode *newtree = tr_reroot2(tree, mp);
      tree = newtree;
    }
    return tree;
  }
}

void nj_update_seq_to_node_map(TreeNode *tree, char **names, CovarData *data) {
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
  int *seq_idx = nj_build_seq_idx(leaves, names);
  
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
   this will be done as a side effect of nj_points_to_distances but upon initialization
   it needs to be done separately */
void nj_update_diam_leaves(Matrix *D, CovarData *data) {
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

void nj_repair_zero_br(TreeNode *t) {
  for (int nodeidx = 0; nodeidx < lst_size(t->nodes); nodeidx++) {
    TreeNode *n = lst_get_ptr(t->nodes, nodeidx);
    if (n->parent != NULL && n->dparent <= 0)
      n->dparent = 1e-3;
  }
}
