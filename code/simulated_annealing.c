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
  double *h;

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
    if (env->samples->dimension) {
            printf("Best solution fouAFDSASFADFSASDFnd:\n");

    }

  if (h != NULL) {
    printf("Best solution found as sadfasdfasdf:\n");

    // Loop through the elements of the array
    for (int i = 0; i < 5; i++) {
        printf("%f ", h[i]);
    }
    printf("\n");

    // Once you are done with 'best_solution', don't forget to free the memory
    // free(best_solution);
} else {
    printf("Error: best_solution is NULL\n");
}
      printf("Best solution poiupouipouiupoi:\n");

  N = env_p->samples->dimension;  // Set the value of N
      printf("Best solution fouAFDSASFADFSASDFnd:\n");

  moving_avg = calloc(N, sizeof(double));
  prev_change = calloc(N, sizeof(double));
  srand48(*seed);
      printf("Best solution 23232323:\n");

  if(!h0) {
    //initialize to all zeros
    //this could be best random hyperplane instead
    //h0 = CALLOC(env->samples->dimension+1, double);
    h0 = best_random_hyperplane_unbiased(1, env);
  }
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());
      printf("Best solution gsgdfgsdfgdsgsf:\n");

  gsl_siman_solve(r, h0, energy, siman_step_cosine, hplane_dist, print_hplane, NULL, NULL, NULL, (env->samples->dimension)*sizeof(double), p);

  double *random_solution = blank_solution(env->samples);
  double random_objective_value = hyperplane_to_solution(h0, random_solution, env);
  printf("obj = %g\n", random_objective_value);
  free(moving_avg);
  free(prev_change);
  return random_solution;
}


// Genetic Algorithm

#define POP_SIZE 1000
#define MUTATION_RATE 0.01
#define CROSSOVER_RATE 0.7
#define MAX_GENERATIONS 1000

gsl_rng *r;

double fitness(double *h) {
    double value = hyperplane_to_solution(h, NULL, env);
    
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

#define TABU_TENURE 1
#define NEIGHBORHOOD_SIZE 1
#define MAX_ITERATIONS 1

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
        if (!env) {
            printf("Best solution ===========================:\n");

    }
    N = env->samples->dimension;
    if (!h0)
    {
        h0 = best_random_hyperplane_unbiased(1, env);
    }
    r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, *seed);

    double current_solution[env->samples->dimension];
    memcpy(current_solution, h0, (env->samples->dimension) * sizeof(double));

    // Allocate memory on the heap for best_solution
    double *best_solution = malloc(env->samples->dimension * sizeof(double));
    if (!best_solution) {
        // Handle memory allocation failure
        printf("Memory allocation failed for best_solution\n");
        gsl_rng_free(r);
        return NULL;
    }
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
    gsl_rng_free(r);
    h = best_solution;
    h = single_siman_run_param(0, 0, env, NULL, params); // todo

    return best_solution; // Return the heap-allocated array
}



// Particle Swarm

// #define SWARM_SIZE 30
// #define MAX_ITERATIONS 1000
// #define W 0.5  // inertia weight
// #define C1 1.5  // cognitive/personal weight
// #define C2 1.5  // social weight

#define SWARM_SIZE 300
#define MAX_ITERATIONS 10000
#define W 0.8  // inertia weight
#define C1 1.8  // cognitive/personal weight
#define C2 1.8  // social weight


gsl_rng *r;
env_t *env;

typedef struct {
    double *position;
    double *velocity;
    double *pbest;
    double pbest_value;
} Particle;

double pso_objective(double *h) {
    return -hyperplane_to_solution(h, NULL, env);
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
