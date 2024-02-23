#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_siman.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <float.h>
#include <stdbool.h>  // <-- Include this for bool, true, and false


#define N_TRIES 800 /* how many points do we try before stepping */
#define ITERS_FIXED_T 2000 /* how many iterations for each T? */
#define K 3.0 /* Boltzmann constant */
#define STEP_SIZE 1.0 /* max step size in random walk */
#define T_INITIAL 1.0 /* initial temperature */
#define MU_T 1.003 /* damping factor for temperature */
#define T_MIN 0.1 // NOTE: changed

gsl_siman_params_t params = {N_TRIES, ITERS_FIXED_T, STEP_SIZE, K, T_INITIAL, MU_T, T_MIN};
env_t *env; //this probably could be included in the current state xp
gsl_rng *r;

int N;  // Global variable to store n
double *moving_avg;  // Moving average array
double *prev_change;

// WIP: Must validate correctness
double obj_new(double *h, env_t *env) {
  size_t reach = 0, nfpos = 0;
  size_t dimension = env->samples->dimension;

  for (size_t class = 0; class < env->samples->class_cnt; class++) {
    for (size_t i = 0; i < env->samples->count[class]; i++) {
      // Calculate the dot product between h and the data sample.
      double dot_product = 0;
      for (size_t j = 0; j < dimension; j++) {
        dot_product += h[j] * env->samples->samples[class][i][j];
      }

      if (dot_product >= 1e-15) { // TODO: this threshold might be adjusted
        if (class == 1) {
          reach++;
        } else {
          nfpos++;
        }
      }
    }
  }

  double prec;
  if (reach + nfpos == 0) {
    prec = 0;
  } else {
    prec = ((double)reach) / (reach + nfpos);
  }

  double violation = (env->params->theta - 1) * reach + env->params->theta * nfpos;
  if (violation < 0) {
    violation = 0;
  }

  return reach - violation * env->params->lambda;
}

double energy(void *xp) {
  //returns -1 * obj of hyperplane specified in xp
  //this will be minimized
  double *h = xp;
      // printf("%f %f %f\n", h[0], h[1], h[2]);
  // printf("h = %f\n", *h);
//   double obj = hyperplane_to_solution(h, NULL, env);
  double obj = obj_new(h, env);
// printf("obj = %f\n", obj);
  // printf("h = %f, obj = %f\n", *h, obj);
  return -obj;
}

double* orthogonal_to(const double *a, const double *b, int n) {
  double *result = (double *)calloc(n, sizeof(double));
  
  // Calculate dot products
  double dot_a_b = 0.0;
  double dot_a_a = 0.0;
  for (int i = 0; i < n; i++) {
    dot_a_b += a[i] * b[i];
    dot_a_a += a[i] * a[i];
  }

  for (int i = 0; i < n; i++) {
    result[i] = b[i] - (dot_a_b / dot_a_a) * a[i];
  }

  // Normalizing the result
  double norm = 0.0;
  for (int i = 0; i < n; i++) {
    norm += result[i] * result[i];
  }
  norm = sqrt(norm);
  
  for (int i = 0; i < n; i++) {
    result[i] /= norm;
  }
  
  return result;
}


void siman_step(const gsl_rng *r, void *xp, double step_size) {
  int n = env->samples->dimension;
  double *h = (double *) xp;
  
  // Get a random unit vector u
  double *u = CALLOC(n, double);
  gsl_ran_dir_nd(r, n, u); // random unit vector

  // Get a vector w that's orthogonal to both h and u
  double *w = orthogonal_to(h, u, n);

  for (int i = 0; i < n; i++)
  {
      double newhi = h[i] + step_size * w[i];
      h[i] = newhi;
  }

  double norm = 0.0;
  for (int i = 0; i < n; i++)
  {
      norm += h[i] * h[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < n; i++)
  {
      h[i] /= norm;
  }

  free(u);
  free(w);
}

double dot_product(const double *a, const double *b, int n) {
  double dot = 0.0;
  for (int i = 0; i < n; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

void siman_step_ortho(const gsl_rng *r, void *xp, double step_size) {
  int n = env->samples->dimension;
  double *h = (double *)xp;
  
  // Get a random unit vector u orthogonal to h
  double *u = (double *)calloc(n, sizeof(double));
  // todo change it cones. algorithm
  do {
    gsl_ran_dir_nd(r, n, u);
  } while(fabs(dot_product(h, u, n)) < 1e-10); // ensure u is not parallel to h

  // Rotate h in the plane defined by h and u by step_size radians
  // First, project h onto the orthogonal plane
  double *perpendicular = orthogonal_to(u, h, n);
  
  // Apply the rotation
  double cos_angle = cos(step_size);
  double sin_angle = sin(step_size);
  for (int i = 0; i < n; ++i) {
      h[i] = cos_angle * perpendicular[i] + sin_angle * u[i];
  }

  // Normalize the resulting vector to ensure it's a unit vector
  double norm_h = 0.0;
  for (int i = 0; i < n; ++i) {
      norm_h += h[i] * h[i];
  }
  norm_h = sqrt(norm_h);
  for (int i = 0; i < n; ++i) {
      h[i] /= norm_h;
  }

  free(u);
  free(perpendicular);
}

// Function to create a random 2D rotation matrix within an n-dimensional space
gsl_matrix *random_2d_rotation_matrix(const gsl_rng *r, int n, double angle) {
    gsl_matrix *R = gsl_matrix_alloc(n, n);
    gsl_matrix_set_identity(R);

    int i = gsl_rng_uniform_int(r, n);
    int j = gsl_rng_uniform_int(r, n);
    while (i == j) {  // Ensure i and j are different for a valid 2D plane
        j = gsl_rng_uniform_int(r, n);
    }

    // Set the 2D rotation components within the larger matrix
    gsl_matrix_set(R, i, i, cos(angle));
    gsl_matrix_set(R, i, j, -sin(angle));
    gsl_matrix_set(R, j, i, sin(angle));
    gsl_matrix_set(R, j, j, cos(angle));

    return R;
}

// Function to rotate the hyperplane normal vector 'h' by 'step_size' degrees
void siman_rotate_step(const gsl_rng *r, void *xp, double step_size) {
    int n = env->samples->dimension;  // Assuming 'env' and 'samples' are defined and accessible
    double *h = (double *) xp;
    // double angle = step_size * M_PI / 180.0;  // Convert step size from degrees to radians
  double angle = step_size * gsl_rng_uniform(r) * 2.0 * M_PI; // Random angle between 0 and 2*PI

    // Print the hyperplane normal vector before rotation
    // printf("Hyperplane normal vector before rotation:\n");
    // for (int i = 0; i < n; ++i) {
    //     printf("%g ", h[i]);
    // }
    // printf("\n");

    // Create a random 2D rotation matrix within the n-dimensional space
    gsl_matrix *R = random_2d_rotation_matrix(r, n, angle);

    // Create a GSL vector that views the data in 'h'
    gsl_vector_view h_view = gsl_vector_view_array(h, n);

    // Create a temporary GSL vector to hold the result
    gsl_vector *temp_result = gsl_vector_alloc(n);

    // Multiply the normal vector 'h' by the rotation matrix and store the result in 'temp_result'
    gsl_blas_dgemv(CblasNoTrans, 1.0, R, &h_view.vector, 0.0, temp_result);

    // Copy the result from 'temp_result' back to 'h'
    gsl_vector_memcpy(&h_view.vector, temp_result);

    // Print the hyperplane normal vector after rotation
    // printf("Hyperplane normal vector after rotation:\n");
    // for (int i = 0; i < n; ++i) {
    //     printf("%g ", h[i]);
    // }
    // printf("\n");

    // Clean up
    gsl_vector_free(temp_result);
    gsl_matrix_free(R);
}

void siman_step_raw(const gsl_rng *r, void *xp, double step_size) {
    int n = env->samples->dimension;
    double *h = (double *)xp;

    // 1. Find a random vector on the hyperplane
    double *random_vector = (double *)calloc(n, sizeof(double));
    gsl_ran_dir_nd(r, n, random_vector);

    // 2. Normalize the random vector's magnitude
    double norm_random_vector = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_random_vector += random_vector[i] * random_vector[i];
    }
    norm_random_vector = sqrt(norm_random_vector);
    for (int i = 0; i < n; ++i) {
        random_vector[i] /= norm_random_vector;
    }

    // 3. Add it to the hyperplane
    for (int i = 0; i < n; ++i) {
        h[i] += random_vector[i] * step_size;
    }

    // 4. Normalize the hyperplane
    double norm_h = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_h += h[i] * h[i];
    }
    norm_h = sqrt(norm_h);
    for (int i = 0; i < n; ++i) {
        h[i] /= norm_h;
    }

    free(random_vector);
}

void siman_step_cosine(const gsl_rng *r, void *xp, double step_size) {
    int n = env->samples->dimension;
    double *h = (double *)xp;

    // Compute the unit vector parallel to h
    double *u = (double *)calloc(n, sizeof(double));
    double norm_h = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_h += h[i] * h[i];
    }
    norm_h = sqrt(norm_h);
    for (int i = 0; i < n; ++i) {
        u[i] = h[i] / norm_h;
    }

    // Generate a random vector
    double *random_vector = (double *)calloc(n, sizeof(double));
    gsl_ran_dir_nd(r, n, random_vector);

    // Compute a vector perpendicular to h
    double *uperp = (double *)calloc(n, sizeof(double));
    double dot_product = 0.0;
    for (int i = 0; i < n; ++i) {
        dot_product += random_vector[i] * u[i];
    }
    for (int i = 0; i < n; ++i) {
        uperp[i] = random_vector[i] - dot_product * u[i];
    }

    // Normalize uperp
    double norm_uperp = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_uperp += uperp[i] * uperp[i];
    }
    norm_uperp = sqrt(norm_uperp);
    for (int i = 0; i < n; ++i) {
        uperp[i] /= norm_uperp;
    }

    // Form the new vector h
    double cos_theta = cos(step_size);
    double sin_theta = sin(step_size);
    for (int i = 0; i < n; ++i) {
        h[i] = cos_theta * u[i] + sin_theta * uperp[i];
    }

    // Normalize the resulting vector h
    norm_h = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_h += h[i] * h[i];
    }
    norm_h = sqrt(norm_h);
    for (int i = 0; i < n; ++i) {
        h[i] /= norm_h;
    }

    free(u);
    free(random_vector);
    free(uperp);
}

double calculate_tan(const gsl_vector *a, const gsl_vector *b) {
    double dot_product;
    gsl_blas_ddot(a, b, &dot_product);  // Compute the dot product of a and b

    // Compute the norms of a and b
    double norm_a = gsl_blas_dnrm2(a);
    double norm_b = gsl_blas_dnrm2(b);

    // Compute the cosine of the angle
    double cos_theta = dot_product / (norm_a * norm_b);

    // Ensure the cosine is not zero to prevent division by zero
    if (cos_theta == 0) {
        fprintf(stderr, "Vectors are orthogonal. Tangent is undefined.\n");
        return HUGE_VAL;  // Return a large number to indicate an undefined tangent
    }

    // Compute the sine of the angle using the Pythagorean identity
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Compute the tangent of the angle
    double tan_theta = sin_theta / cos_theta;

    return tan_theta;
}

// Function to create a Givens rotation matrix
gsl_matrix *create_givens_rotation(int n, int i, int j, double angle) {
    gsl_matrix *G = gsl_matrix_alloc(n, n);
    if (!G) {
        fprintf(stderr, "Failed to allocate memory for matrix G\n");
        exit(EXIT_FAILURE);
    }
    gsl_matrix_set_identity(G);
    gsl_matrix_set(G, i, i, cos(angle));
    gsl_matrix_set(G, i, j, -sin(angle));
    gsl_matrix_set(G, j, i, sin(angle));
    gsl_matrix_set(G, j, j, cos(angle));
    return G;
}

// The Aguilera-Perez Algorithm for computing the rotation matrix
gsl_matrix *aguilera_perez_rotation_matrix(const gsl_vector *v, int n) {
    gsl_matrix *M = gsl_matrix_alloc(n, n);
    if (!M) {
        fprintf(stderr, "Failed to allocate memory for matrix M\n");
        exit(EXIT_FAILURE);
    }
    gsl_matrix_set_identity(M);

    for (int r = 2; r <= n-1; ++r) {
        for (int c = n; c >= r; --c) {
            double tan_angle = calculate_tan(v, v); // Placeholder for actual computation
            double angle = atan(tan_angle);
            gsl_matrix *G = create_givens_rotation(n, c-1, c, angle);

            // Update M by multiplying with G
            gsl_matrix *temp_M = gsl_matrix_alloc(n, n);
            if (!temp_M) {
                fprintf(stderr, "Failed to allocate memory for temporary matrix temp_M\n");
                gsl_matrix_free(M);
                gsl_matrix_free(G);
                exit(EXIT_FAILURE);
            }
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, M, G, 0.0, temp_M);
            gsl_matrix_memcpy(M, temp_M);

            gsl_matrix_free(G);
            gsl_matrix_free(temp_M);
        }
    }

    return M;
}

// Function to rotate the hyperplane normal vector 'h' by 'step_size' degrees
void siman_rotate_step_paper(const gsl_rng *r, void *xp, double step_size) {
    int n = env->samples->dimension;  // Using the placeholder 'env' variable
    double *h = (double *) xp;
    double angle = step_size * gsl_rng_uniform(r) * 2.0 * M_PI; // Random angle between 0 and 2*PI

    // Create a random 2D rotation matrix within the n-dimensional space
    gsl_matrix *R = random_2d_rotation_matrix(r, n, angle);

    // Create a GSL vector that views the data in 'h'
    gsl_vector_view h_view = gsl_vector_view_array(h, n);

    // Create a temporary GSL vector to hold the result
    gsl_vector *temp_result = gsl_vector_alloc(n);
    if (!temp_result) {
        fprintf(stderr, "Failed to allocate memory for vector temp_result\n");
        gsl_matrix_free(R);
        exit(EXIT_FAILURE);
    }

    // Multiply the normal vector 'h' by the rotation matrix and store the result in 'temp_result'
    gsl_blas_dgemv(CblasNoTrans, 1.0, R, &h_view.vector, 0.0, temp_result);

    // Copy the result from 'temp_result' back to 'h'
    gsl_vector_memcpy(&h_view.vector, temp_result);

    // Clean up
    gsl_vector_free(temp_result);
    gsl_matrix_free(R);
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

double hplane_dist_spatial(void *xp, void *yp) {
  double *h1 = xp, *h2 = yp;
  int n = env->samples->dimension;
  gsl_vector x1 = gsl_vector_view_array(h1, n).vector;
  gsl_vector x2 = gsl_vector_view_array(h2, n).vector;
  gsl_vector_sub(&x1, &x2); //x1 -= x2
  double dist = gsl_blas_dnrm2(&x1);
  // printf("h1 = %p, h2 = %p, dist = %f\n", (void *)h1, (void *)h2, dist);
  return dist;
}


double hplane_dist(void *xp, void *yp) {
    double *h1 = xp, *h2 = yp;
    int n = env->samples->dimension;

    gsl_vector x1 = gsl_vector_view_array(h1, n).vector;
    gsl_vector x2 = gsl_vector_view_array(h2, n).vector;

    // Normalize the vectors
    // gsl_vector_scale(&x1, 1.0 / gsl_blas_dnrm2(&x1));
    // gsl_vector_scale(&x2, 1.0 / gsl_blas_dnrm2(&x2));

    // Calculate the dot product between x1 and x2.
    double dot_product = 0;
    gsl_blas_ddot(&x1, &x2, &dot_product);

    // Clamp dot_product to [-1, 1] to avoid potential floating point issues.
    if (dot_product > 1.0) dot_product = 1.0;
    if (dot_product < -1.0) dot_product = -1.0;

    // Calculate the inverse cosine of the dot product.
    double angular_distance = acos(dot_product);

    return angular_distance;
}

void print_hplane(void *xp) {
  // printf ("%12g", *((double *) xp));
  // double *h = xp;
  // int n = env->samples->dimension;
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
    h0 = best_random_hyperplane_unbiased(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, h0, energy, siman_step_cosine, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension)*sizeof(double), params);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  return random_solution;
}

double *single_siman_run_param(unsigned int *seed, int iter_lim, env_t *env_p, double *h0, gsl_siman_params_t p) {
  env = env_p;
  N = env->samples->dimension;  // Set the value of N
  moving_avg = calloc(N, sizeof(double));
  prev_change = calloc(N, sizeof(double));
  srand48(*seed);
  if(!h0) {
    //initialize to all zeros
    //this could be best random hyperplane instead
    //h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane_unbiased(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());

  gsl_siman_solve(r, h0, energy, siman_step_cosine, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension)*sizeof(double), p);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  free(moving_avg);
  free(prev_change);
  return random_solution;
}


// Genetic Algorithm

#define POP_SIZE 300
#define MUTATION_RATE 1.2
#define CROSSOVER_RATE 0.1
#define MAX_GENERATIONS 400

gsl_rng *r;

double fitness(double *h) {
    double value = obj_new(h, env);
    
    if (isnan(value) || isinf(value)) {
        return DBL_MAX;  // Assign a large penalty for invalid values
    }
    
    return -value;  // We negate to minimize
}

void mutate(double *h) {
    int n = env->samples->dimension;
    for (int i = 0; i < n; i++) {
        if (gsl_rng_uniform(r) < MUTATION_RATE) {
            double mutation_value = (gsl_rng_uniform(r) * 2.0 - 1.0) * 0.1;  // Reduced mutation step
            h[i] += mutation_value;
        }
    }
}

void crossover(double *parent1, double *parent2, double *child) {
    int n = env->samples->dimension;
    for (int i = 0; i < n; i++) {
        if (gsl_rng_uniform(r) < CROSSOVER_RATE) {
            child[i] = parent1[i];
        } else {
            child[i] = parent2[i];
        }
    }
}

double* genetic_algorithm_run(unsigned int *seed, env_t *env_p) {
    env = env_p;
    r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, *seed);

    int n = env->samples->dimension;
    double population[POP_SIZE][n];
    double new_population[POP_SIZE][n];
    double fitness_values[POP_SIZE];

    // Initialize population
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < n; j++) {
            population[i][j] = gsl_rng_uniform(r) * 2.0 - 1.0;  // Random value between -1 and 1
        }
    }

    double best_fitness = DBL_MAX;  // Initialize with a large value

    for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
        // Evaluate fitness
        for (int i = 0; i < POP_SIZE; i++) {
            fitness_values[i] = fitness(population[i]);
            if (fitness_values[i] < best_fitness) {
                best_fitness = fitness_values[i];
            }
        }

        // Create new population
        for (int i = 0; i < POP_SIZE; i+=2) {
            // Simple tournament selection
            double *parent1 = population[gsl_rng_uniform_int(r, POP_SIZE)];
            double *parent2 = population[gsl_rng_uniform_int(r, POP_SIZE)];
            if (fitness(parent2) > fitness(parent1)) {
                double *temp = parent1;
                parent1 = parent2;
                parent2 = temp;
            }

            crossover(parent1, parent2, new_population[i]);
            crossover(parent1, parent2, new_population[i+1]);
            mutate(new_population[i]);
            mutate(new_population[i+1]);
        }

        // Replace old population
        for (int i = 0; i < POP_SIZE; i++) {
            for (int j = 0; j < n; j++) {
                population[i][j] = new_population[i][j];
            }
        }
    }

    printf("Best Fitness: %f\n", -best_fitness);  // Negate to get the actual value
    double *best_solution = NULL;
    for (int i = 0; i < POP_SIZE; i++) {
        if (fitness(population[i]) == best_fitness) {
            best_solution = population[i];
            break;
        }
    }
    return best_solution;
}




// tabu Search Algorithm

// #define TABU_TENURE 5
// #define NEIGHBORHOOD_SIZE 10
// #define MAX_ITERATIONS 1000

#define TABU_TENURE 10
#define NEIGHBORHOOD_SIZE 300
#define MAX_ITERATIONS 1500

double tabu_objective(double *h) {
    return -obj_new(h, env);
}

void generate_neighbor(double *h, double *neighbor) {
    int n = env->samples->dimension;
    for (int i = 0; i < n; i++) {
        neighbor[i] = h[i] + (gsl_rng_uniform(r) * 2.0 - 1.0) * 0.1;  // Small random perturbation
    }
}

double *tabu_search_run(unsigned int *seed, env_t *env_p, double *h0)
{
    env = env_p;
    if (!h0)
    {
        // initialize to all zeros
        // this could be best random hyperplane instead
        // h0 = CALLOC(env->samples->dimension+1, double);
        h0 = best_random_hyperplane_unbiased(1, env);
    }
    r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, *seed);

    double current_solution[env->samples->dimension];
    memcpy(current_solution, h0, (env->samples->dimension) * sizeof(double));

    double best_solution[env->samples->dimension];
    memcpy(best_solution, h0, (env->samples->dimension) * sizeof(double));

    double best_value = tabu_objective(best_solution);

    // Tabu list
    double tabu_list[TABU_TENURE][env->samples->dimension];
    int tabu_index = 0;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        double best_neighbor[env->samples->dimension];
        double best_neighbor_value = DBL_MAX;

        for (int i = 0; i < NEIGHBORHOOD_SIZE; i++) {
            double neighbor[env->samples->dimension];
            generate_neighbor(current_solution, neighbor);
            double neighbor_value = tabu_objective(neighbor);

            // Check if neighbor is better and not in tabu list
            bool is_tabu = false;
            for (int j = 0; j < TABU_TENURE; j++) {
                if (memcmp(neighbor, tabu_list[j], (env->samples->dimension) * sizeof(double)) == 0) {
                    is_tabu = true;
                    break;
                }
            }

            if (!is_tabu && neighbor_value < best_neighbor_value) {
                best_neighbor_value = neighbor_value;
                memcpy(best_neighbor, neighbor, (env->samples->dimension) * sizeof(double));
            }
        }

        // Update current solution and best solution
        memcpy(current_solution, best_neighbor, (env->samples->dimension) * sizeof(double));
        if (best_neighbor_value < best_value) {
            best_value = best_neighbor_value;
            memcpy(best_solution, best_neighbor, (env->samples->dimension) * sizeof(double));
        }

        // Update tabu list
        memcpy(tabu_list[tabu_index], best_neighbor, (env->samples->dimension) * sizeof(double));
        tabu_index = (tabu_index + 1) % TABU_TENURE;
    }

    printf("Best Objective Value: %f\n", best_value);
    return best_solution;
}



// Particle Swarm

// #define SWARM_SIZE 30
// #define MAX_ITERATIONS 1000
// #define W 0.5  // inertia weight
// #define C1 1.5  // cognitive/personal weight
// #define C2 1.5  // social weight

#define SWARM_SIZE 600
#define MAX_ITERATIONS 1000
#define W 0.7  // inertia weight
#define C1 1.5  // cognitive/personal weight
#define C2 1.9  // social weight


gsl_rng *r;
env_t *env;

typedef struct {
    double *position;
    double *velocity;
    double *pbest;
    double pbest_value;
} Particle;

double pso_objective(double *h) {
    return -obj_new(h, env);
}

void update_velocity_and_position(Particle *particle, double *gbest, int dimension) {
    for (int i = 0; i < dimension; i++) {
        double r1 = gsl_rng_uniform(r);
        double r2 = gsl_rng_uniform(r);
        particle->velocity[i] = W * particle->velocity[i] + C1 * r1 * (particle->pbest[i] - particle->position[i]) + C2 * r2 * (gbest[i] - particle->position[i]);
        particle->position[i] += particle->velocity[i];
    }
}

double* pso_run(unsigned int *seed, env_t *env_p) {
    env = env_p;
    r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, *seed);

    int dimension = env->samples->dimension;

    Particle swarm[SWARM_SIZE];
    double gbest[dimension];
    double gbest_value = DBL_MAX;

    // Initialize swarm
    for (int i = 0; i < SWARM_SIZE; i++) {
        swarm[i].position = malloc(dimension * sizeof(double));
        swarm[i].velocity = malloc(dimension * sizeof(double));
        swarm[i].pbest = malloc(dimension * sizeof(double));

        for (int j = 0; j < dimension; j++) {
            swarm[i].position[j] = gsl_rng_uniform(r) * 2.0 - 1.0;
            swarm[i].velocity[j] = gsl_rng_uniform(r) * 2.0 - 1.0;
        }

        swarm[i].pbest_value = pso_objective(swarm[i].position);
        memcpy(swarm[i].pbest, swarm[i].position, dimension * sizeof(double));

        if (swarm[i].pbest_value < gbest_value) {
            gbest_value = swarm[i].pbest_value;
            memcpy(gbest, swarm[i].position, dimension * sizeof(double));
        }
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < SWARM_SIZE; i++) {
            update_velocity_and_position(&swarm[i], gbest, dimension);

            double current_value = pso_objective(swarm[i].position);
            if (current_value < swarm[i].pbest_value) {
                swarm[i].pbest_value = current_value;
                memcpy(swarm[i].pbest, swarm[i].position, dimension * sizeof(double));
            }

            if (current_value < gbest_value) {
                gbest_value = current_value;
                memcpy(gbest, swarm[i].position, dimension * sizeof(double));
            }
        }
    }

    printf("Best Objective Value: %f\n", gbest_value);

    // Cleanup
    for (int i = 0; i < SWARM_SIZE; i++) {
        free(swarm[i].position);
        free(swarm[i].velocity);
        free(swarm[i].pbest);
    }

    return gbest;
}
