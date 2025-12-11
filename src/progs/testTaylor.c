#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "phast/tree_model.h"
#include "phast/trees.h"
#include "phast/vector.h"
#include "phast/matrix.h"
#include "phast/stacks.h"
#include "phast/misc.h"
#include "geometry.h"
#include "mvn.h"
#include "multi_mvn.h"
#include "phast/tree_model.h"
#include "taylor.h"   /* your TaylorData, CovarData, typedefs */
#include "likelihoods.h"
#include "backprop.h"
/**********************************************************************
 * UTILITIES
 **********************************************************************/

static double randu() {
  return (double)rand() / (double)RAND_MAX;
}

static int almost_equal(double a, double b, double tol) {
  return fabs(a - b) < tol;
}

static void require(int cond, const char *msg) {
  if (!cond) {   
    fprintf(stderr, "[FAIL] %s\n", msg);
    /* exit(1); */
    assert(0);
  }
}

static inline void vec_inc(Vector *v, int idx, double a) {
    vec_set(v, idx, vec_get(v, idx) + a);
}

/* Build a simple binary PHAST tree using a Newick string. */
static TreeNode *make_binary_test_tree(int nseqs)
{
    if (nseqs < 2)
        die("make_binary_test_tree: nseqs must be >= 2\n");

    /* Build a Newick string bottom-up */
    List *subtrees = stk_new_ptr(nseqs);

    for (int i = 0; i < nseqs; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "taxon%d:1", i+1);
        lst_push_ptr(subtrees, copy_charstr(buf));
    }

    while (lst_size(subtrees) > 1) {
        char *A = stk_pop_ptr(subtrees);
        char *B = stk_pop_ptr(subtrees);

        char buf[256];
        snprintf(buf, sizeof(buf), "(%s,%s):1", A, B);

        sfree(A);
        sfree(B);

        lst_push_ptr(subtrees, copy_charstr(buf));
    }

    char *final = stk_pop_ptr(subtrees);
    lst_free(subtrees);

    char newick[512];
    snprintf(newick, sizeof(newick), "%s", final);
    sfree(final);

    return tr_new_from_string(newick);
}

static TreeModel *make_fake_tree_model(int nseqs)
{
    /* Build a proper resolved binary tree */
    TreeNode *tree = make_binary_test_tree(nseqs);

    /* Dummy reversible model: alphabet = "A", 1 state, trivial rate matrix */
    MarkovMatrix *Q = mm_new(1, "A", CONTINUOUS);
    Vector *bg = vec_new(1); vec_set(bg, 0, 1.0);

    TreeModel *tm = tm_new(tree,
                           Q,                  /* rate matrix */
                           bg,                 /* background freqs */
                           REV,                /* subst model */
                           "A",                /* alphabet */
                           1,                  /* rate cats */
                           1.0,                /* alpha */
                           NULL,               /* rate consts */
                           -1);

    return tm;
}



CovarData *make_fake_covar_data(int nseqs, int dim)
{
  Matrix *dist = mat_new(nseqs, nseqs);
  for (int i = 0; i < nseqs; i++)
    for (int j = 0; j < nseqs; j++)
      mat_set(dist, i, j, (i == j ? 0.0 : 1.0));

  /* unique dummy names */
  char **names = smalloc(nseqs * sizeof(char *));
  for (int i = 0; i < nseqs; i++) {
    names[i] = smalloc(16);
    sprintf(names[i], "taxon_%d", i);
  }

  CovarData *cd =
    nj_new_covar_data(
        DIAG,
        dist,
        dim,
        NULL, NULL,
        names,
        0,        /* natural_grad */
        1.0,      /* kld_upweight */
        -1,
        0.0,      /* var_reg */
        0,        /* hyperbolic */
        1.0,      /* negcurv */
        0,        /* ultrametric */
        0,        /* radial_flow */
        0,        /* planar_flow */
        NULL, NULL,
        1         /* use_taylor */
    );

  /* Minimal MVN: μ = 0, Σ = I */
  multi_MVN *mmvn = mmvn_new(nseqs, dim, MVN_DIAG);
  int nx = nseqs * dim;
  for (int i = 0; i < nx; i++) {
    vec_set(mmvn->mvn->mu, i, 0.0);
    mat_set(mmvn->mvn->sigma, i, i, 1.0);
  }

  cd->taylor->mmvn = mmvn;
  
  cd->taylor->mod = make_fake_tree_model(nseqs);


  return cd;
}

/* save current branch lengths from tree into vector bl */
static inline
void tr_save_branch_lengths(TreeNode *root, Vector *bl) {
  assert(root->nnodes == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    vec_set(bl, i, n->dparent);
  }
}


/* Build a simple 4-taxon NJ tree using the real nj_infer_tree()
   This replaces tr_new_star_tree() in the tests */
static TreeNode *make_test_tree(int nseqs)
{
  /* For now support only nseqs == 4 as in the unit tests */
  if (nseqs != 4)
    die("make_test_tree: only nseqs=4 supported in this test.\n");

  /* Create a very simple distance matrix */
  Matrix *D = mat_new(4,4);
  mat_zero(D);

  /*
    Taxon layout:
    0--\
    +-- internal --\
    1--/                +-- root
    /
    2--\               /
    +-- internal--/
    3--/
     
    Use symmetric distances that NJ will handle deterministically
  */

  double dist[4][4] = {
    {0, 2, 5, 5},
    {2, 0, 5, 5},
    {5, 5, 0, 2},
    {5, 5, 2, 0}
  };

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      mat_set(D, i, j, dist[i][j]);

  /* simple names */
  char *names[4] = {"A", "B", "C", "D"};

  /* Use the real NJ inference */
  TreeNode *root = nj_infer_tree(D, names, NULL, NULL);

  mat_free(D);
  return root;
}


/**********************************************************************
 * CREATE A TOY MODEL FOR TESTING J AND Jᵀ
 * (Tree, embeddings, Jacobians)
 **********************************************************************/

static TreeModel *make_toy_model(int nseqs) {
  /* Create an extremely small fake model:
     - a tree with nseqs leaves, star-shaped
     - branch lengths all = 1
     - No flows, Euclidean only
  */
  TreeModel *m = smalloc(sizeof(TreeModel));
  m->tree = make_test_tree(nseqs);

  /* no CRISPR, no hyperbolic */
  m->tree->nnodes = nseqs + 1; /* if star tree */
  return m;
}

static TaylorData *make_toy_tay(int nseqs, int dim, CovarData *cd) {
  TaylorData *td = smalloc(sizeof(TaylorData));
  td->covar_data = cd;
  td->nseqs = nseqs;
  td->dim = dim;
  td->fulld = nseqs * dim;
  td->nbranches = nseqs;   /* star tree: n leaves = n branches */
  td->ndist = (nseqs * (nseqs - 1)) / 2;
  
  /* allocate Jbx, JbxT later in tay_prep_jacobians */
  td->Jbx = NULL;
  td->JbxT = NULL;

  /* workspace */
  td->tmp_x1 = vec_new(td->fulld);
  td->tmp_x2 = vec_new(td->fulld);
  td->tmp_dD = vec_new((nseqs * (nseqs - 1)) / 2);
  td->tmp_dy = vec_new(td->fulld);
  td->tmp_extra = vec_new(td->fulld);

  /* Set y = the mean embedding = zeros for simplicity */
  td->y = vec_new(td->fulld);
  vec_zero(td->y);

  return td;
}

/**********************************************************************
 * 1. TEST SIGMA–VECTOR MULTIPLICATION
 **********************************************************************/

static void test_sigma_vec_mult_CONST() {
  printf("Test CONST Σv ...\n");

  int n = 4, d = 1;
  multi_MVN *mmvn = mmvn_new(n, d, MVN_DIAG);
  CovarData *cd = smalloc(sizeof(CovarData));
  cd->type = CONST;
  cd->lambda = 2.5;

  Vector *v = vec_new(n);
  Vector *out = vec_new(n);

  for (int i=0;i<n;i++) vec_set(v, i, i + 1.0);

  tay_sigma_vec_mult(out, mmvn, v, cd);

  for (int i=0;i<n;i++) {
    double want = cd->lambda * vec_get(v, i);
    require(almost_equal(vec_get(out,i), want, 1e-12),
            "CONST Σv mismatch");
  }

  vec_free(v); vec_free(out);
  free(cd);
  mmvn_free(mmvn);
  printf("  OK\n");
}

static void test_sigma_vec_mult_DIAG() {
  printf("Test DIAG Σv ...\n");

  int n = 4, d = 1;
  multi_MVN *mmvn = mmvn_new(n,d,MVN_DIAG);
  CovarData *cd = smalloc(sizeof(CovarData));
  cd->type = DIAG;

  /* set diagΣ = [1,2,3,4] */
  for (int i=0;i<n;i++)
    mat_set(mmvn->mvn->sigma, i,i, i+1.0);

  Vector *v = vec_new(n);
  Vector *out = vec_new(n);
  for (int i=0;i<n;i++) vec_set(v,i, (i+1)*0.5);

  tay_sigma_vec_mult(out, mmvn, v, cd);

  for (int i=0;i<n;i++) {
    double want = (i+1.0) * vec_get(v,i);
    require(almost_equal(vec_get(out,i), want, 1e-12),
            "DIAG Σv mismatch");
  }

  vec_free(v); vec_free(out);
  free(cd);
  mmvn_free(mmvn);
  printf("  OK\n");
}

static void test_sigma_vec_mult_LOWR() {
  printf("Test LOWR Σv ...\n");

  int n = 4, d = 1, r = 2;
  /* Build R explicitly */
  Matrix *R = mat_new(n, r);
  /* Something simple */
  for (int i=0;i<n;i++)
    for (int j=0;j<r;j++)
      mat_set(R,i,j, 0.1*(i+1) + 0.3*(j+1));

  /* mmvn of type MVN_LOWR */
  multi_MVN *mmvn = mmvn_new(n, d, MVN_LOWR);
  mvn_reset_LOWR(mmvn->mvn, R);

  CovarData *cd = smalloc(sizeof(CovarData));
  cd->type = LOWR;

  Vector *v = vec_new(n*d);
  Vector *out = vec_new(n*d);
  Matrix *Sigma = mat_new(n,n);

  /* Sigma = R Rᵀ */
  mat_set_gram(Sigma, R);

  for (int i=0;i<n;i++) vec_set(v,i, randu());

  tay_sigma_vec_mult(out, mmvn, v, cd);

  /* brute-force check: out ≈ Σv */
  for (int i=0;i<n;i++) {
    double sum = 0.0;
    for (int j=0;j<n;j++)
      sum += mat_get(Sigma, i,j) * vec_get(v,j);

    require(almost_equal(vec_get(out,i), sum, 1e-12),
            "LOWR Σv mismatch");
  }

  vec_free(v); vec_free(out);
  mat_free(Sigma);
  mat_free(R);
  free(cd);
  mmvn_free(mmvn);

  printf("  OK\n");
}

/* Finite-difference test of Jbx = d b / d x
   using the dt_dD Jacobian from nj_infer_tree.

   We do:
     x      -> D(x) (pairwise distances)
     D(x)   -> b    (branch lengths), with fixed neighbor tape
   with analytic  Jbx from tay_prep_jacobians,
   and finite-diff Jbx ≈ (dt_dD * (D_eps - D0)/eps).
*/
static void test_tay_Jfun_fd()
{
  printf("Testing tay_prep_jacobians (Jbx) via finite-difference ...\n");

  int n   = 4;
  int dim = 2;
  int nx  = n * dim;
  int nb  = 2 * n - 2;                     /* matches tay_new() */

  /* --- set up CovarData and multi_MVN --- */
  CovarData *cd = make_fake_covar_data(n, dim);
  TaylorData *td = cd->taylor;

  /* --- build a mean embedding x --- */
  Vector *x = vec_new(nx);
  for (int i = 0; i < nx; i++)
    vec_set(x, i, 0.1 * i + 0.05);        /* simple, non-degenerate */

  /* store x as the mean embedding in TaylorData */
  vec_copy(td->y, x);                     /* td->y used by tay_dx_from_dt */

  /* --- compute distances at mean --- */
  nj_points_to_distances(x, cd);          /* fills cd->dist */

  /* --- build NJ tree WITH dt_dD and neighbor tape --- */
  int npairs = n * (n - 1) / 2;
  Matrix *dt_dD = mat_new(nb, npairs);    /* Jacobian db/dD at mean */

  TreeNode *root =
    nj_infer_tree(cd->dist, cd->names, dt_dD, td->nb);

  /* --- build a TreeModel around that tree --- */
  TreeModel *mod = make_fake_tree_model(n);  /* your existing helper */
  nj_reset_tree_model(mod, root);           /* attach root & branch lengths */
  td->mod = mod;                             /* needed by tay_dx_from_dt */

  /* --- build analytic Jbx via tay_prep_jacobians --- */
  tay_prep_jacobians(td, mod, x);           /* fills td->Jbx, td->JbxT */

  /* --- precompute D0 in vectorized form --- */
  Vector *D0 = vec_new(npairs);
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      int idx = nj_i_j_to_dist(i, j, n);
      vec_set(D0, idx, mat_get(cd->dist, i, j));
    }
  }

  /* --- workspace for finite-diff --- */
  Vector *x_eps  = vec_new(nx);
  Matrix *dist_eps = mat_new(n, n);
  Vector *D_eps  = vec_new(npairs);
  Vector *dD     = vec_new(npairs);        /* (D_eps - D0)/eps */
  Vector *b_col  = vec_new(nb);            /* dt_dD * dD */

  double eps = 1e-6;

  /* --- test each column j of Jbx --- */
  for (int j = 0; j < nx; j++) {

    /* x_eps = x; x_eps[j] += eps */
    vec_copy(x_eps, x);
    vec_set(x_eps, j, vec_get(x_eps, j) + eps);

    /* recompute distances at x_eps into dist_eps */
    nj_points_to_distances(x_eps, cd);
    mat_copy(dist_eps, cd->dist);

    /* vectorize D_eps */
    for (int i = 0; i < n; i++) {
      for (int k = i+1; k < n; k++) {
        int idx = nj_i_j_to_dist(i, k, n);
        vec_set(D_eps, idx, mat_get(dist_eps, i, k));
      }
    }

    /* dD = (D_eps - D0)/eps */
    vec_copy(dD, D_eps);
    vec_minus_eq(dD, D0);
    vec_scale(dD, 1.0 / eps);

    /* b_col = dt_dD * dD (finite-diff approximation to column j of Jbx) */
    mat_vec_mult(b_col, dt_dD, dD);

    /* compare to analytic Jbx[:, j] */
    for (int i = 0; i < nb; i++) {
      double Jij = mat_get(td->Jbx, i, j);
      double fd  = vec_get(b_col, i);

      /* tolerance can be fairly loose; these are second-order objects */
      if (!almost_equal(Jij, fd, 1e-4)) {
        fprintf(stderr,
                "Mismatch in Jbx at (branch=%d, coord=%d): "
                "analytic=%g, fd=%g, diff=%g\n",
                i, j, Jij, fd, Jij - fd);
        require(FALSE, "tay_prep_jacobians finite-diff check failed");
      }
    }
  }

  vec_free(x);
  vec_free(x_eps);
  vec_free(D0);
  vec_free(D_eps);
  vec_free(dD);
  vec_free(b_col);
  mat_free(dist_eps);
  mat_free(dt_dD);

  /* free tree model, etc., as appropriate for your test harness */

  printf("  tay_prep_jacobians finite-diff test: OK\n");
}

/**********************************************************************
 * 3. TEST S v BY EXPLICIT S = J Σ Jᵀ
 **********************************************************************/

static void test_SVP_against_explicit() {
  printf("Test S v = (J Σ Jᵀ) v ...\n");

  int n = 4, d = 1;
  int fulld = n*d;

  CovarData *cd = smalloc(sizeof(CovarData));
  cd->type = CONST;
  cd->lambda = 2.0;
  cd->hyperbolic = 0;

  TreeModel *mod = make_toy_model(n);
  TaylorData *td = make_toy_tay(n, d, cd);
  td->mod = mod;

  Vector *x = td->y;   /* zero mean */
  tay_prep_jacobians(td, mod, x);

  /* Build explicit S = J Σ Jᵀ */
  Matrix *S = mat_new(td->nbranches, td->nbranches);

  /* Σ = lambda * I for CONST case */
  for (int i=0;i<td->nbranches;i++)
    for (int j=0;j<td->nbranches;j++) {
      /* S(i,j) = J(i,:) (λ I) J(j,:)ᵀ = λ * J(i,:)·J(j,:) */
      double sum = 0.0;
      for (int k=0;k<fulld;k++)
        sum += mat_get(td->Jbx, i,k) * mat_get(td->Jbx, j,k);
      mat_set(S,i,j, cd->lambda * sum);
    }

  Vector *v = vec_new(td->nbranches);
  Vector *out = vec_new(td->nbranches);
  Vector *want = vec_new(td->nbranches);

  for (int i=0;i<v->size;i++) vec_set(v,i, randu());

  tay_SVP(out, v, td);

  /* explicit multiply want = S v */
  for (int i=0;i<S->nrows;i++) {
    double sum = 0.0;
    for (int j=0;j<S->ncols;j++)
      sum += mat_get(S,i,j) * vec_get(v,j);
    vec_set(want,i,sum);
  }

  for (int i=0;i<v->size;i++) {
    require(almost_equal(vec_get(out,i), vec_get(want,i), 1e-10),
            "S v mismatch");
  }

  vec_free(v); vec_free(out); vec_free(want);
  mat_free(S);

  printf("  OK\n");
}

/**********************************************************************
 * FINITE DIFFERENCE HELPERS
 **********************************************************************/

/* Finite-difference directional check:
   out ≈ (F(x + eps*v) - F(x))/eps
   where F: R^n → R^n is a vector function (e.g., gradient)
*/
static void fd_directional(Vector *fd,           /* output */
                           void (*F)(Vector*, Vector*, void*),
                           Vector *x,
                           Vector *v,
                           double eps,
                           void *aux)
{
  int n = x->size;
  Vector *xplus = vec_new(n);
  Vector *Fx = vec_new(n);
  Vector *Fxplus = vec_new(n);

  vec_copy(xplus, x);
  for (int i = 0; i < xplus->size; i++)
    vec_set(xplus, i, vec_get(xplus, i) + eps * vec_get(v, i));

  F(Fx, x, aux);
  F(Fxplus, xplus, aux);

  for (int i=0;i<n;i++)
    vec_set(fd, i, (vec_get(Fxplus,i) - vec_get(Fx,i)) / eps);

  vec_free(xplus);
  vec_free(Fx);
  vec_free(Fxplus);
}

/**********************************************************************
 * TOY QUADRATIC FOR HVP TESTING
 **********************************************************************/

typedef struct {
  Matrix *A;     /* symmetric n×n */
} ToyAux;

/* grad f(x) = A x */
static void toy_grad(Vector *out, Vector *x, void *aux)
{
  ToyAux *ta = (ToyAux*)aux;
  mat_vec_mult(out, ta->A, x);
}

/* Hessian–vector product: H v = A v  */
static void toy_HVP(Vector *out, Vector *v, void *aux)
{
  ToyAux *ta = (ToyAux*)aux;
  mat_vec_mult(out, ta->A, v);
}

/**********************************************************************
 * TEST SIGMA_GRAD_MULT USING FINITE DIFFERENCES
 **********************************************************************/
static void test_sigma_grad_mult_fd()
{
  printf("Test tay_sigma_grad_mult via finite differences ...\n");

  double eps = 1e-6;
  int n = 4, d = 1;
  int nx = n * d;

  /* ------------------------------------------------------------
     (1) Build trivial distance matrix (required by CovarData)
     ------------------------------------------------------------ */
  Matrix *dist = mat_new(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mat_set(dist, i, j, (i == j ? 0.0 : 1.0));

  /* dummy names */
  char **names = smalloc(n * sizeof(char *));
  for (int i = 0; i < n; i++)
    names[i] = "taxon";

  /* ------------------------------------------------------------
     (2) Properly construct CovarData using your real constructor
     ------------------------------------------------------------ */
  CovarData *cd =
      nj_new_covar_data(
          DIAG,        /* covariance type             */
          dist,        /* distance matrix             */
          d,           /* embedding dimension         */
          NULL, NULL,  /* no MSA, no CRISPR           */
          names,
          0,           /* natural_grad                */
          1.0,         /* kld_upweight                */
          -1,          /* rank (unused for DIAG)      */
          0.0,         /* var_reg                     */
          0,           /* hyperbolic                  */
          1.0,         /* negcurvature (ignored)      */
          0,           /* ultrametric                 */
          0,           /* radial_flow                 */
          0,           /* planar_flow                 */
          NULL,        /* TreePrior                   */
          NULL,        /* MigTable                    */
          1            /* use_taylor = TRUE           */
      );

  /* now cd->taylor exists and is valid */

  /* ------------------------------------------------------------
     (3) Build minimal multi_MVN of matching dimension
     ------------------------------------------------------------ */
  multi_MVN *mmvn = mmvn_new(n, d, MVN_DIAG);
  cd->taylor->mmvn = mmvn;   /* hook mmvn into TaylorData */

  /* initialize sigma to identity */
  for (int i = 0; i < nx; i++)
    mat_set(mmvn->mvn->sigma, i, i, 1.0);

  /* ensure covariance is updated from cd->params */
  nj_update_covariance(mmvn, cd);

  /* ------------------------------------------------------------
     (4) Allocate test vectors
     ------------------------------------------------------------ */
  Vector *v     = vec_new(nx);
  Vector *Sv0   = vec_new(nx);
  Vector *Sv1   = vec_new(nx);
  Vector *fd    = vec_new(nx);
  Vector *gradp = vec_new(nx);

  for (int i = 0; i < nx; i++)
    vec_set(v, i, randu());

  /* ------------------------------------------------------------
     (5) Loop over each parameter p and test ∂Σ/∂θ_p * v
     ------------------------------------------------------------ */
  for (int p = 0; p < n; p++) {

    /* analytic gradient */
    tay_sigma_grad_mult(gradp, v, mmvn, cd);

    /* baseline Σ v */
    tay_sigma_vec_mult(Sv0, mmvn, v, cd);

    /* finite difference perturbation */
    double old = vec_get(cd->params, p);
    vec_set(cd->params, p, old + eps);

    nj_update_covariance(mmvn, cd);
    tay_sigma_vec_mult(Sv1, mmvn, v, cd);

    /* restore */
    vec_set(cd->params, p, old);
    nj_update_covariance(mmvn, cd);

    /* fd = (Sv1 - Sv0)/eps */
    for (int i = 0; i < nx; i++) {
      double diff = (vec_get(Sv1, i) - vec_get(Sv0, i)) / eps;
      vec_set(fd, i, diff);
    }

    /* Compare analytic vs finite difference */
    for (int i = 0; i < nx; i++) {
      require(
          almost_equal(vec_get(fd, i), vec_get(gradp, i), 1e-5),
          "Sigma_grad mismatch in finite difference test");
    }
  }

  /* ------------------------------------------------------------
     (6) Free memory
     ------------------------------------------------------------ */
  vec_free(v);
  vec_free(Sv0);
  vec_free(Sv1);
  vec_free(fd);
  vec_free(gradp);

  mmvn_free(mmvn);
  /* CovarData owns dist and names, but since this is a test: */
  mat_free(dist);
  free(names);
  /* TaylorData is owned by cd and should be freed later */
  free(cd);

  printf("  OK\n");
}

/**********************************************************************
 * FINITE DIFFERENCE TEST OF HVP LOGIC
 **********************************************************************/

static void test_HVP_fd() {
  printf("Test HVP finite-difference logic ...\n");

  int n = 6;
  double eps = 1e-5;

  /* Build random symmetric A */
  Matrix *A = mat_new(n,n);
  for (int i=0;i<n;i++)
    for (int j=i;j<n;j++) {
      double v = randu();
      mat_set(A,i,j,v);
      mat_set(A,j,i,v);
    }

  ToyAux aux;
  aux.A = A;

  Vector *x = vec_new(n);
  Vector *v = vec_new(n);

  for (int i=0;i<n;i++) {
    vec_set(x,i, randu());
    vec_set(v,i, randu());
  }

  Vector *Hvp = vec_new(n);   /* analytic = A v */
  Vector *fd = vec_new(n);    /* finite diff */

  /* analytic */
  toy_HVP(Hvp, v, &aux);

  /* finite difference: fd = (g(x+eps v) - g(x))/eps */
  fd_directional(fd, toy_grad, x, v, eps, &aux);

  /* Compare */
  for (int i=0;i<n;i++) {
    require(almost_equal(vec_get(Hvp,i), vec_get(fd,i), 1e-5),
            "HVP finite-difference mismatch");
  }

  vec_free(x);
  vec_free(v);
  vec_free(Hvp);
  vec_free(fd);
  mat_free(A);

  printf("  OK\n");
}

/**********************************************************************
 * MINI-LIKELIHOOD TEST USING REAL NJ DISTANCES + BRANCH LENGTHS
 *
 * ℓ(x) = Σ_{i<j} (D_ij(x) − T_ij)^2
 *
 * This gives:
 *   - real dℓ/dx  (via autodiff-free analytic expression)
 *   - real HVP via tay_HVP
 *   - real finite-difference HVP
 *
 * Uses your existing distance computation + NJ tree fit.
 **********************************************************************/

typedef struct {
  CovarData *data;
  TreeModel *mod;
  Matrix    *targetD;    /* fixed target distances */
  Vector    *workspace_grad; /* size n*d */
} MiniLikAux;

/* Compute mini-likelihood gradient dℓ/dx */
static void mini_lik_grad(Vector *out, Vector *x, void *aux)
{
  MiniLikAux *A = (MiniLikAux*)aux;
  CovarData *data = A->data;
  int n = data->nseqs;
  int dim = data->dim;

  /* Update distances and tree from x */
  nj_points_to_distances(x, data);
  TreeNode *tree = nj_inf(data->dist, data->names, NULL, NULL, data);
  nj_reset_tree_model(A->mod, tree);

  /* ℓ = Σ (D_ij - T_ij)^2 → gradient wrt D_ij → wrt x */
  vec_zero(out);

  Vector *dL_dD = vec_new(n*(n-1)/2);
  Vector *dL_dy = vec_new(n*dim);

  vec_zero(dL_dD);

  for (int i=0;i<n;i++)
    for (int j=i+1;j<n;j++) {
      int idx = nj_i_j_to_dist(i,j,n);
      double Dij = mat_get(data->dist,i,j);
      double Tij = mat_get(A->targetD,i,j);
      double w = 2.0*(Dij - Tij);
      vec_set(dL_dD, idx, w);
    }

  /* Distances → embedding via your code path; no flows here */
  vec_zero(dL_dy);
  for (int i=0;i<n;i++) {
    int base_i = i*dim;
    for (int j=i+1;j<n;j++) {
      int base_j = j*dim;
      int idx = nj_i_j_to_dist(i,j,n);

      double Dij = mat_get(data->dist,i,j);
      if (Dij < 1e-15) Dij = 1e-15;
      double w = vec_get(dL_dD, idx);

      for (int d=0; d<dim; d++) {
        double diff = vec_get(x, base_i+d) - vec_get(x, base_j+d);
        double g = w * diff / (Dij * data->pointscale * data->pointscale);
        vec_inc(dL_dy, base_i+d, g);
        vec_inc(dL_dy, base_j+d, -g);
      }
    }
  }

  vec_copy(out, dL_dy);

  vec_free(dL_dD);
  vec_free(dL_dy);
}

/* HVP wrapper: v → (H mini_lik) v using your tay_HVP */
static void mini_lik_HVP(Vector *out, Vector *v, void *aux)
{
  MiniLikAux *A = (MiniLikAux*)aux;

  /* reuse your HVP, assuming A->data->taylor is prepopulated */
  tay_HVP(out, v, A->data->taylor);
}

/**********************************************************************
 * Test mini-likelihood HVP via finite difference
 **********************************************************************/
static void test_HVP_mini_lik_fd()
{
  printf("Mini-likelihood NJ HVP test ...\n");

  int n = 3, dim = 2;
  int nx = n * dim;

  /* Build synthetic CovarData + TreeModel */
  CovarData *cd = make_fake_covar_data(n, dim);  /* you already have a helper for this */
  TreeModel *mod = smalloc(sizeof(TreeModel));
  mod->tree = NULL;   /* caller will set using NJ below */

  /* Build target distances T_ij */
  Matrix *T = mat_new(n,n);
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++)
      mat_set(T,i,j, 1.0 + 0.3*(i+j));  /* arbitrary symmetric */

  MiniLikAux aux;
  aux.data = cd;
  aux.mod = mod;
  aux.targetD = T;
  aux.workspace_grad = vec_new(nx);

  /* random embedding x */
  Vector *x = vec_new(nx);
  for (int i=0;i<nx;i++) vec_set(x,i, randu());

  /* v */
  Vector *v = vec_new(nx);
  for (int i=0;i<nx;i++) vec_set(v,i, randu());

  /* analytic HVP */
  Vector *Hvp = vec_new(nx);
  mini_lik_HVP(Hvp, v, &aux);

  /* finite difference */
  Vector *fd = vec_new(nx);
  fd_directional(fd, mini_lik_grad, x, v, 1e-6, &aux);

  /* compare */
  for (int i=0;i<nx;i++)
    require(almost_equal(vec_get(Hvp,i), vec_get(fd,i), 1e-4),
            "Mini-likelihood HVP finite-difference mismatch");

  vec_free(Hvp);
  vec_free(fd);
  vec_free(x);
  vec_free(v);
  vec_free(aux.workspace_grad);
  mat_free(T);

  printf("  OK\n");
}

/**********************************************************************
 * Test tay_prep_jacobians via finite-difference
 **********************************************************************/
static void test_tay_prep_jacobians_fd()
{
  printf("Test tay_prep_jacobians (finite difference) ...\n");

  int n = 4, dim = 2;
  int nx = n * dim;

  CovarData *cd = make_fake_covar_data(n, dim);
  TreeModel *mod = smalloc(sizeof(TreeModel));
  mod->tree = NULL;   /* caller will set using NJ below */

  /* Mean embedding x */
  Vector *x = vec_new(nx);
  for (int i=0;i<nx;i++) vec_set(x,i, randu());
  cd->taylor->y = x;
  cd->taylor->dim = dim;
  cd->taylor->nseqs = n;

  /* Update distances + tree */
  nj_points_to_distances(x, cd);
  TreeNode *t = nj_inf(cd->dist, cd->names, NULL, cd->taylor->nb, cd);
  nj_reset_tree_model(mod, t);

  /* Build true branch vector */
  int nb = mod->tree->nnodes;
  Vector *b = vec_new(nb);
  tr_save_branch_lengths(mod->tree, b);

  /* Build J via code */
  tay_prep_jacobians(cd->taylor, mod, x);

  /* Finite-difference check */
  double eps = 1e-6;
  Vector *xplus = vec_new(nx);
  Vector *bplus = vec_new(nb);

  for (int j=0;j<nx;j++) {

    /* xplus = x + eps e_j */
    vec_copy(xplus, x);
    vec_set(xplus, j, vec_get(xplus,j) + eps);

    /* recompute tree for perturbed x */
    nj_points_to_distances(xplus, cd);
    TreeNode *t2 = nj_inf(cd->dist, cd->names, NULL, NULL, cd);
    nj_reset_tree_model(mod, t2);

    tr_save_branch_lengths(mod->tree, bplus);

    /* for each branch i compare: J[i,j] vs. (bplus-b)/eps */
    for (int i=0;i<nb;i++) {
      double fd = (vec_get(bplus,i) - vec_get(b,i)) / eps;
      double Jij = mat_get(cd->taylor->Jbx, i, j);
      require(almost_equal(fd, Jij, 1e-4),
              "Jbx finite-difference mismatch");
    }
  }

  vec_free(x);
  vec_free(xplus);
  vec_free(b);
  vec_free(bplus);

  printf("  OK\n");
}

/**********************************************************************
 * MAIN
 **********************************************************************/

int main() {
  srand(1);

  printf("=== Taylor Micro Tests ===\n\n");

  test_sigma_vec_mult_CONST();
  test_sigma_vec_mult_DIAG();
  test_sigma_vec_mult_LOWR();
  test_tay_Jfun_fd();
  /* test_SVP_against_explicit(); */

  test_sigma_grad_mult_fd();
  test_HVP_fd();

  test_HVP_mini_lik_fd();
  test_tay_prep_jacobians_fd();

  printf("\nAll tests PASSED.\n");
  return 0;
}
