#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>

int print_matrix(FILE *f, const gsl_matrix *m) {
  int status, n = 0;

  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      if ((status = fprintf(f, "%g ", gsl_matrix_get(m, i, j))) < 0)
        return -1;
      n += status;
    }

    if ((status = fprintf(f, "\n")) < 0)
      return -1;
    n += status;
  }

  return n;
}

sys7_t *generate_fixedpt_mat(env_t *env, double *rel_sol) {
  //generates the matrix A and vector b corresponding to system (7), normalized to the form Ax <= b
  int idx_min = idx_extreme(0, 1, 0, env->samples);
  int idx_max = violation_idx(0, env->samples) - 1;
  sparse_vector_t *fixed = sparse_vector_blank(positives(env->samples)+negatives(env->samples));
  int nfixed = 0;
  for(int i = idx_min; i <= idx_max; i++) {
    //if(rel_sol[i-1] > 1) printf("AAA\n");
    if(rel_sol[i-1] == index_label(i, env->samples)) { //off by one because gurobi uses zero-indexing
      if(rel_sol[i-1] == 1e+101) continue;
      if(rel_sol[i-1] > 1) continue;
      append(fixed, i, rel_sol[i-1]);
      if(rel_sol[i-1] != 0 && rel_sol[i-1] != 1) {
	printf("Invalid relaxation value %g\n", rel_sol[i-1]);
	printf("index label = %d\n", index_label(i, env->samples));
	printf("equal? %d\n", rel_sol[i-1] == index_label(i, env->samples));
	printf("Invalid relaxation value %g\n", rel_sol[i-1]);
	printf("index label = %d\n", index_label(i, env->samples));

	printf("i = %d\n", i);
	printf("What? %d\n", 1e+101==1);
      }
      nfixed++;
    }
  }

  gsl_matrix *A = gsl_matrix_alloc(nfixed, env->samples->dimension + 1);
  gsl_vector *b = gsl_vector_alloc(nfixed);
  //gsl_vector_set_all(b, 3.14159);
  int *ref = CALLOC(nfixed, int);
  gsl_vector *row = gsl_vector_alloc(env->samples->dimension + 1);
  for(int k = 1; k <= fixed->len; k++) {
    int i = fixed->ind[k];
    double val = fixed->val[k];
    ref[k-1] = i;
    sample_locator_t *loc = locator(i, env->samples);
    if(val == 1) {
      for(int j = 0; j < env->samples->dimension; j++) {
	//value should be -si[j]
	gsl_vector_set(row, j, -(env->samples->samples[loc->class][loc->index][j]));
      }
      gsl_vector_set(row, env->samples->dimension, 1); //coefficient of c
      gsl_vector_set(b, k-1, -env->params->epsilon_positive);
    } else if(val == 0) {
      for(int j = 0; j < env->samples->dimension; j++) {
	//value should be si[j]
	//NOTE: I had this as -si[j] in the code. just changed to +
	gsl_vector_set(row, j, (env->samples->samples[loc->class][loc->index][j]));
      }
      gsl_vector_set(row, env->samples->dimension, -1);
      gsl_vector_set(b, k-1, -env->params->epsilon_negative);
    } else {
      printf("invalid fixed value %g. i = %d, k = %d\n", val, i, k);
      exit(0);
    }
    gsl_matrix_set_row(A, k-1, row);
    free(loc);
  }
  gsl_vector_free(row);
  free(delete_sparse_vector(fixed));
  
  sys7_t *out = malloc(sizeof(sys7_t));
  out->A = A;
  out->b = b;
  out->ref = ref;

  return out;
}

gsl_vector *rand_unit_ball(gsl_rng *rng, int d) {
  //returns random point in unit ball in Rd
  double *surf = malloc(d*sizeof(double)); //random point on surface of ball
  gsl_ran_dir_nd(rng, d, surf);
  double r = pow(gsl_rng_uniform(rng), 1.0/d); //weighted probability of choosing the radius
  gsl_vector *x = gsl_vector_alloc(d);
  for(int i = 0; i < d; i++)
    gsl_vector_set(x, i, surf[i]*r);
  free(surf);
  return x;
}

double square(double x) {
  return x*x;
}

gsl_matrix *hessian(sys7_t *s, gsl_vector *x) {
  gsl_matrix *A = s->A;
  gsl_vector *b = s->b;
  gsl_matrix *H = gsl_matrix_calloc(A->size2, A->size2);
  for(int i = 0; i < A->size1; i++) {
    gsl_vector ai = gsl_matrix_row(A, i).vector;
    gsl_matrix_get_row(&ai, A, i);
    double aiTx;
    gsl_blas_ddot(x, &ai, &aiTx);
    double denom = square(gsl_vector_get(b, i) - aiTx);
    gsl_matrix aiM = gsl_matrix_view_vector(&ai, ai.size, 1).matrix;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1/denom, &aiM, &aiM, 1, H);
  }
  return H;
}

double dikin_norm(sys7_t *s, gsl_vector *z, gsl_vector *x) {
  //compute ||z-x||_x = (z-x)^T H(x) (z-x)
  int n = z->size;
  gsl_vector *diff = gsl_vector_alloc(n);
  gsl_vector_memcpy(diff, z);
  gsl_vector_sub(diff, x); //diff = z - x
  gsl_matrix *H = hessian(s, x);

  gsl_vector *Hzx = gsl_vector_alloc(n);
  gsl_blas_dgemv(CblasNoTrans, 1, H, diff, 0, Hzx);

  double res;
  gsl_blas_ddot(diff, Hzx, &res);
  
  gsl_vector_free(diff);
  gsl_matrix_free(H);
  gsl_vector_free(Hzx);

  return res;
}

void dikin_step(sys7_t *s, gsl_vector *x) {
  /* this function will change x(i) into x(i+1)*/
  //if(rand() > RAND_MAX/2) return;
  gsl_matrix *H = hessian(s, x);
  int n = H->size1;
  gsl_eigen_symmv_workspace *ws = gsl_eigen_symmv_alloc(n);
  gsl_vector *evals = gsl_vector_alloc(n);
  gsl_matrix *E = gsl_matrix_alloc(n, n);
  gsl_eigen_symmv(H, evals, E, ws);

  gsl_matrix *L = gsl_matrix_alloc(n, n); //matrix with new eigenvalues along main diagonal

  for(int i = 0; i < n; i++) {
    double *li = gsl_vector_ptr(evals, i);
    double *Lii = gsl_matrix_ptr(L, i, i);
    //set li := li^(-1/2) = 1/sqrt(li)
    *li = 1/(sqrt(*li));
    *Lii = *li;
  }

  gsl_matrix *EL = gsl_matrix_alloc(n, n);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, E, L, 0, EL);

  gsl_matrix *M = gsl_matrix_alloc(n, n); //M = H^(-1/2) = ELE^(-1) = ELE^T since E is an orthogonal matrix
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, EL, E, 0, M);

  gsl_eigen_symmv_free(ws);
  gsl_vector_free(evals);
  gsl_matrix_free(E);
  gsl_matrix_free(L);
  gsl_matrix_free(EL);

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, rand());

  double r = 0.5;
  gsl_vector *y = gsl_vector_alloc(n);      

  char rnd_method = 'u'; //u for uniform, g for gaussian
  if(rnd_method == 'g') {
    gsl_vector *g = gsl_vector_alloc(n);
    for(int i = 0; i < n; i++) {
      gsl_vector_set(g, i, gsl_ran_gaussian(rng, 1));
    }
    
    gsl_vector_memcpy(y, x);
    gsl_blas_dgemv(CblasNoTrans, r/sqrt(n), M, g, 1, y); //y := x + (r/sqrt(n))Mg
    gsl_vector_free(g);
  } else if(rnd_method == 'u') {
    gsl_vector *u = rand_unit_ball(rng, n);

    gsl_vector_memcpy(y, x);
    gsl_blas_dgemv(CblasNoTrans, pow(r, n), M, u, 1, y);
    
    gsl_vector_free(u);
  } else return;
  
  if(dikin_norm(s, x, y) <= r) { //if x in Dy, accept probabilistically
    double prob, detHy, detHx;
    int signumx, signumy;
    gsl_permutation *px = gsl_permutation_alloc(n);
    gsl_permutation *py = gsl_permutation_alloc(n);
    gsl_linalg_LU_decomp(H, px, &signumx);
    gsl_matrix *Hy = hessian(s, y);
    gsl_linalg_LU_decomp(Hy, py, &signumy);
    detHx = gsl_linalg_LU_det(H, signumx);
    detHy = gsl_linalg_LU_det(Hy, signumy);
    prob = sqrt(detHy/detHx);
    if(gsl_rng_uniform(rng) < prob) {
      gsl_vector_memcpy(x, y); //x := y
    }
    gsl_matrix_free(Hy);
    gsl_permutation_free(px);
    gsl_permutation_free(py);
  }

  gsl_vector_free(y);
  gsl_rng_free(rng);
  gsl_matrix_free(H);
}

double *random_constrained_hyperplane(env_t *env, double *rel_sol, sys7_t *s) {
  //printf("rel_sol[49] = %g\n", rel_sol[49]);
  //s = generate_fixedpt_mat(env, rel_sol);
  
  gsl_vector *x = gsl_vector_alloc(env->samples->dimension+1);
  for(int i = 0; i <= env->samples->dimension; i++)
    gsl_vector_set(x, i, rel_sol[i]);

  //verify that x satisfies system
  gsl_vector *Ax = gsl_vector_alloc(s->b->size);

  int viol = 0;
  gsl_blas_dgemv(CblasNoTrans, 1, s->A, x, 0, Ax);
  /*for(int i = 0; i < Ax->size; i++) {
    if(gsl_vector_get(Ax, i) - gsl_vector_get(s->b, i) > 0.1) {
      printf("constraint violation. i = %d, Axi = %g, bi = %g\n", i, gsl_vector_get(Ax, i), gsl_vector_get(s->b, i));
      sample_locator_t *loc = locator(s->ref[i], env->samples);
      printf("class = %d, rel val = %g, xi = ", loc->class, rel_sol[s->ref[i]]);
      for(int j = 0; j < env->samples->dimension; j++) {
	printf("%g ", env->samples->samples[loc->class][loc->index][j]);
      }
      printf("\n");
      free(loc);
      viol = 1;
    }
    }*/
  /*  if(viol) {
    printf("Verifying constraint 4c for i = 0\n");
    sample_locator_t *loc = locator(s->ref[0], env->samples);
    printf("xs = %g\n", rel_sol[s->ref[0]]);
    double sTw = 0;
    for(int j = 0; j < env->samples->dimension; j++)
      sTw += env->samples->samples[loc->class][loc->index][j]*rel_sol[j];
    printf("1 + sTw - c - eps = %g\n", 1+sTw-rel_sol[env->samples->dimension] - env->params->epsilon_positive);
    free(loc);
    exit(0);
    
    printf("A = \n");
    print_matrix(stdout, s->A);
    printf("x = \n");
    gsl_matrix xm = gsl_matrix_view_vector(x, x->size, 1).matrix;
    print_matrix(stdout, &xm);
    printf("Ax = \n");
    gsl_matrix Axm = gsl_matrix_view_vector(Ax, Ax->size, 1).matrix;
    print_matrix(stdout, &Axm);
    printf("b = \n");
    gsl_matrix bm = gsl_matrix_view_vector(s->b, s->b->size, 1).matrix;
    print_matrix(stdout, &bm);
    exit(0);
    }*/
  
  dikin_step(s, x);

  double *h = CALLOC(env->samples->dimension+1, double);
  for(int i = 0; i <= env->samples->dimension; i++) {
    h[i] = gsl_vector_get(x, i);
  }
  return h;
}
