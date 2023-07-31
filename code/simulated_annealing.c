#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_siman.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>


// #define N_TRIES 200 /* how many points do we try before stepping */
#define N_TRIES 800 /* how many points do we try before stepping */
// #define N_TRIES 3200 /* how many points do we try before stepping */

// #define ITERS_FIXED_T 500 /* how many iterations for each T? */
#define ITERS_FIXED_T 2000 /* how many iterations for each T? */
// #define ITERS_FIXED_T 8000 /* how many iterations for each T? */

// #define K 1.0 /* Boltzmann constant */
#define K 3.0 /* Boltzmann constant */
// #define K 10.0 /* Boltzmann constant */

#define STEP_SIZE 1.0 /* max step size in random walk */
// #define STEP_SIZE 10000.0 /* max step size in random walk */
// #define STEP_SIZE 100000000.0 /* max step size in random walk */



#define T_INITIAL 1.0 /* initial temperature */
#define MU_T 1.003 /* damping factor for temperature */
#define T_MIN 0.1 // NOTE: changed

gsl_siman_params_t params = {N_TRIES, ITERS_FIXED_T, STEP_SIZE, K, T_INITIAL, MU_T, T_MIN};
env_t *env; //this probably could be included in the current state xp

// this is the function that is not working
double energy(void *xp) {
  //returns -1 * obj of hyperplane specified in xp
  //this will be minimized
  double *h = xp;
      // printf("%f %f %f\n", h[0], h[1], h[2]);
  // printf("h = %f\n", *h);
  double obj = hyperplane_to_solution(h, NULL, env);
// printf("obj = %f\n", obj);
  // printf("h = %f, obj = %f\n", *h, obj);
  return -obj;
}

// void take_step(const gsl_rng *r, void *xp, double step_size) {
//     double *hyperplane = (double *)xp;

//     for (int i = 0; i < N; i++) {
//         hyperplane[i] += gsl_ran_gaussian(r, step_size);
//     }
// }

void siman_step(const gsl_rng *r, void *xp, double step_size) {
  int n = env->samples->dimension + 1; //add 1 to include c in the space
  double *h = (double *) xp;
  double *u = CALLOC(n, double);
  gsl_ran_dir_nd(r, n, u); //random unit vector
  for(int i = 0; i < n; i++) {
    double newhi = h[i] + step_size * u[i];
    h[i] = newhi;
  }
  free(u);
}

double hplane_dist(void *xp, void *yp) {
  double *h1 = xp, *h2 = yp;
  //return the distance between the two corresponding vectors in (d+1)-space
  int n = env->samples->dimension + 1;
  gsl_vector x1 = gsl_vector_view_array(h1, n).vector;
  gsl_vector x2 = gsl_vector_view_array(h2, n).vector;
  gsl_vector_sub(&x1, &x2); //x1 -= x2
  double dist = gsl_blas_dnrm2(&x1);
  // printf("h1 = %p, h2 = %p, dist = %f\n", (void *)h1, (void *)h2, dist);
  return dist;
}

void print_hplane(void *xp) {
  // printf ("%12g", *((double *) xp));
  // double *h = xp;
  // int n = env->samples->dimension + 1;
  // for(int i = 0; i < n; i++)
  //   printf("%g ", h[i]);
  // printf("%g ", h[0]); //placeholder - full plane takes too much space
  // printf("n/a");
}

double *single_siman_run(unsigned int *seed, int iter_lim, env_t *env_p, double *h0) {
  env = env_p;
  srand48(*seed);
  if(!h0) {
    //initialize to all zeros
    //this could be best random hyperplane instead
    //h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, h0, energy, siman_step, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension+1)*sizeof(double), params);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  return random_solution;
}

double *single_siman_run_param(unsigned int *seed, int iter_lim, env_t *env_p, double *h0, gsl_siman_params_t p) {
  env = env_p;
  srand48(*seed);
  if(!h0) {
    //initialize to all zeros
    //this could be best random hyperplane instead
    //h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, h0, energy, siman_step, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension+1)*sizeof(double), p);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  return random_solution;
}
