#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_siman.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>


#define N_TRIES 800 /* how many points do we try before stepping */
#define ITERS_FIXED_T 2000 /* how many iterations for each T? */
#define K 3.0 /* Boltzmann constant */
#define STEP_SIZE 1.0 /* max step size in random walk */
#define T_INITIAL 1.0 /* initial temperature */
#define MU_T 1.003 /* damping factor for temperature */
#define T_MIN 0.1 // NOTE: changed

gsl_siman_params_t params = {N_TRIES, ITERS_FIXED_T, STEP_SIZE, K, T_INITIAL, MU_T, T_MIN};
env_t *env; //this probably could be included in the current state xp

int N;  // Global variable to store n
double *moving_avg;  // Moving average array
double *prev_change;

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

// 1. Original gsl_ran_dir_nd random step function
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

// 2. Step function that uses history average
void take_step_average(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;
    double alpha = 0.9;  // Decay factor for moving average

    for (int i = 0; i < N; i++) {
        double change = gsl_ran_gaussian(r, step_size);
        hyperplane[i] += change;

        // Adjust the moving average
        moving_avg[i] = alpha * moving_avg[i] + (1.0 - alpha) * hyperplane[i];
    }
}

// 3. Step function that uses history average with threshold
void take_step_average_threshold(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;
    double alpha = 0.9;  // Decay factor for moving average
    double threshold = 0.1;

    for (int i = 0; i < N; i++) {
        double scaled_step_size = step_size * (1.0 + threshold * moving_avg[i]);
        double change = gsl_ran_gaussian(r, scaled_step_size);
        hyperplane[i] += change;

        // Adjust the moving average
        moving_avg[i] = alpha * moving_avg[i] + (1.0 - alpha) * hyperplane[i];
    }
}

// 4. Step function that uses history of the momentum
void take_step_momentum(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;
    double momentum_factor = 0.7; // A factor to decide how much of the previous change is used.

    for (int i = 0; i < N; i++) {
        double change = gsl_ran_gaussian(r, step_size) + momentum_factor * prev_change[i];
        hyperplane[i] += change;
        prev_change[i] = change;
    }
}

// 5. Step function that uses Gaussian
void take_step_gaussian(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        hyperplane[i] += gsl_ran_gaussian(r, step_size);
    }
}

//6. Step function that uses uniform Rand
void take_step_uniform(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        hyperplane[i] += gsl_rng_uniform(r) * 2 * step_size - step_size;
    }
}

//7. Step function that uses angle
void take_step_angle(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;
    double angle = (gsl_rng_uniform(r) * 2 - 1) * step_size; // random angle between -step_size and step_size
    
    for (int i = 0; i < N - 1; i++) {
        hyperplane[i] = hyperplane[i] * cos(angle) - hyperplane[i+1] * sin(angle);
    }
    hyperplane[N-1] = hyperplane[N-1] * cos(angle);
}

//8. Step function that uses periodic
void take_step_periodic(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        double change = gsl_ran_gaussian(r, step_size) + step_size * sin(hyperplane[i]);
        hyperplane[i] += change;
    }
}

//9. Step function that considers neighbors
void take_step_neighbor(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        double neighbor_influence = (i == 0 ? 0 : hyperplane[i-1] - hyperplane[i]) + (i == N-1 ? 0 : hyperplane[i+1] - hyperplane[i]);
        double change = gsl_ran_gaussian(r, step_size) + 0.2 * neighbor_influence;
        hyperplane[i] += change;
    }
}

//10. Step function that uses direction change.
void take_step_directional(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        double direction = gsl_rng_uniform(r) > 0.5 ? 1.0 : -1.0;
        hyperplane[i] += direction * step_size;
    }
}

//11. Step function that does artificial noise - high focus on exploaraion
double noise_factor = 0.05; // Magnitude of the noise

void take_step_noise_injection(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        hyperplane[i] += gsl_ran_gaussian(r, step_size) + gsl_ran_cauchy(r, noise_factor);
    }
}

//12.  Step function that uses log
void take_step_log_perturbation(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        hyperplane[i] *= exp(gsl_ran_gaussian(r, step_size));
    }
}

//13. Step function uses larger jumps
double jump_probability = 0.005;
double jump_factor = 10;

void take_step_jump(const gsl_rng *r, void *xp, double step_size) {
    double *hyperplane = (double *)xp;

    for (int i = 0; i < N; i++) {
        if (gsl_rng_uniform(r) < jump_probability) {
            hyperplane[i] += gsl_ran_gaussian(r, step_size * jump_factor);
        } else {
            hyperplane[i] += gsl_ran_gaussian(r, step_size);
        }
    }
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
  N = env->samples->dimension + 1;  // Set the value of N
  moving_avg = calloc(N, sizeof(double));
  prev_change = calloc(N, sizeof(double));
  srand48(*seed);
  if(!h0) {
    //initialize to all zeros
    //this could be best random hyperplane instead
    //h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, h0, energy, take_step_average_threshold, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension+1)*sizeof(double), p);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  free(moving_avg);
  free(prev_change);
  return random_solution;
}
