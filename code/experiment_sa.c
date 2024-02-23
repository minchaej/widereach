#include <stdio.h>
#include <math.h>
#include "widereach.h"
#include "helper.h"
#include <gsl/gsl_siman.h>

#define N_TRIES 200 /* how many points do we try before stepping */
#define ITERS_FIXED_T 200 /* how many iterations for each T? */
#define K 3.0 /* Boltzmann constant */
#define STEP_SIZE 1.0 /* max step size in random walk */
#define T_INITIAL 1.0 /* initial temperature */
#define MU_T 1.003 /* damping factor for temperature */
#define T_MIN 0.1 // NOTE: changed


#define SAMPLE_SEEDS 3
unsigned long int samples_seeds[SAMPLE_SEEDS] = {
    // 2020011234955, // 378
    85287339,       // 412
    // 20200623170005 // 433
};

// 711104006
#define MIP_SEEDS 30
unsigned int mip_seeds[MIP_SEEDS] = {
    964156444, 145943044, 869199209, 499223379, 523437323, 964156444,
    248689460, 115706114, 711104006, 311906069, 205328448, 471055100,
    307531192, 543901355, // 24851720, 704008414, 2921762, 181094221,
                          // 234474543, 782516264, 519948660, 115033019, 205486123, 657145193,
                          // 83898336, 41744843, 153111583, 318522606, 952537249, 298531860
};

unsigned long int validation_seed = 711104006;

#define FACT_MAX 9
unsigned int factorial[FACT_MAX];

void initialize_factorial()
{
  factorial[0] = 1.;
  for (size_t i = 1; i < FACT_MAX; i++)
  {
    factorial[i] = i * factorial[i - 1];
  }
}

double fact(unsigned int n)
{
  return (double)factorial[n];
}

int randomIntegerInRange(int min, int max) {
    // Generate a random integer within the range [min, max]
    return (rand() % (max - min + 1)) + min;
}

void feature_scaling(env_t *env) {
  size_t dim = env->samples->dimension;
  int nsamples = 0;
  for(int c = 0; c < env->samples->class_cnt; c++)
    nsamples += env->samples->count[c];
  double *norms = CALLOC(nsamples, double);
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    for(int d = 0; d < dim; d++) {
      //selected class c, feature d
      for(int i = 0; i < cnt; i++)
	norms[d] += fabs(env->samples->samples[c][i][d]);
    }
  }
  //now that norms have been computed, we go back and scale each feature
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    for(int d = 0; d < dim; d++) {
      if(norms[d] == 0){
	printf("Found norm 0 in feature scaling\n");
	continue;
      }
                  
      for(int i = 0; i < cnt; i++)
	env->samples->samples[c][i][d] /= norms[d];
    }
  }
}

double randomDoubleInRange(double min, double max) {
    // Generate a random double within the range [min, max]
    double scale = rand() / (double)RAND_MAX; // [0, 1)
    return min + scale * (max - min);
}

typedef struct exp_res_t
{
  double reach;
  double prec;
} exp_res_t;

exp_res_t experiment(int param_setting)
{
  initialize_factorial();

  env_t env;
  env.params = params_default();
  /*
   * Simplex: 0.99
   *
   * breast cancer 0.99
   * red wine 0.04
   * white wine 0.1
   * south german credit  .9 (2, 1) (originally said 0.95 here)
   * crop mapping  .99 (76, 0.974359);
   * */
  env.params->theta = 0.99;
  double lambda_factor = 100;
  // env.params->theta = 0.99;
  env.params->branch_target = 0.0;
  env.params->iheur_method = deep;
  int n = 400;
  // env.params->lambda = 100 * (n + 1);
  env.params->rnd_trials = 10000;
  // env.params->rnd_trials_cont = 10;
  env.params->rnd_trials_cont = 0;

  // size_t dimension = param_setting;
  size_t dimension = 32;

  clusters_info_t clusters[2];
  // int n = pow10quick(dimension);
  clusters_info_singleton(clusters, n * .8, dimension);
  clusters_info_t *info = clusters + 1;
  info->dimension = dimension;
  size_t cluster_cnt = info->cluster_cnt = 2;
  info->count = CALLOC(cluster_cnt, size_t);
  info->shift = CALLOC(cluster_cnt, double);
  info->side = CALLOC(cluster_cnt, double);
  info->shift[0] = 0.;
  info->side[0] = 1.;
  info->side[1] = pow(.01, 1. / (double)dimension);
  info->shift[1] = 1. - info->side[1];

  info->count[0] = info->count[1] = n / 10;

  double side = sqrt(fact(dimension) / fact(FACT_MAX - 1));
  size_t cluster_sizes[] = {n / 40, n / 80, n / 80, n / 20};
  simplex_info_t simplex_info = {
      .count = n,
      .positives = n / 5,
      .cluster_cnt = 4,
      .dimension = dimension,
      .side = side,
      .cluster_sizes = cluster_sizes};

  srand48(validation_seed);
  samples_t *samples_validation;
  samples_validation = random_samples(n, n / 2, dimension);

  double *h;
  dimension = samples_validation->dimension;
  int solution_size = dimension + samples_total(samples_validation) + 3;
  double *solution = CALLOC(solution_size, double);

  int ntests = SAMPLE_SEEDS * MIP_SEEDS;

  unsigned int *reaches = CALLOC(ntests, unsigned int);
  double *precisions = CALLOC(ntests, double);
  int k = 0;

// seeting k 3 and step3 witl yield reach of 217. <-- old
// new seeting is that since we are using cosine step, we have to make our k and stepsize way smaller
  int sa_param_cnt = 3;
  int sa_n_tries[] = {200, 800, 3200};
  int sa_iters_fixed_t[] = {1000, 2000, 8000};
  double sa_k[] = {1.0, 3.0, 10.0};
  double sa_step_size[] = {0.01, 10000.0, 100000000.0};


  for (int s = 0; s < SAMPLE_SEEDS; s++)
  {
    srand48(samples_seeds[s]);
    printf("Sample seed: %lu\n", samples_seeds[s]);

    samples_t *samples;
    // samples = random_samples(n, n / 2, dimension);

    FILE *infile;

    infile = 
        // fopen("../../data/south-german-credit/SouthGermanCredit.dat", "r");
    	  // fopen("../../data/breast-cancer/wdbc.dat", "r");
        // fopen("../../data/wine-quality/red-cross/winequality-red.dat", "r");
        // fopen("../../data/wine-quality/winequality-white-training.dat", "r"); 
        //  fopen("../../data/crops/small-sample.dat", "r");
          fopen("../../data/crop-classes/small-sample-broadleaf.dat", "r");
        // fopen("../../data/crop-classes/small-sample-canola.dat", "r");


    samples = read_binary_samples(infile);
    fclose(infile);

    env.samples = samples;
    add_bias(samples);
    feature_scaling(&env);
    normalize_samples(samples);
    //print_samples(samples);
    n = samples_total(samples);
    env.params->lambda = lambda_factor * (n + 1);
    env.params->epsilon_precision = 3./990;
    //env.params->epsilon_precision = 3000./990000;

    for (int t = 0; t < MIP_SEEDS; t++)
    {

      unsigned int *seed = mip_seeds + t;

      // grid search...
      for (int i_n_tries = 0; i_n_tries < sa_param_cnt; i_n_tries++)
      {
        for (int i_iters_fixed = 0; i_iters_fixed < sa_param_cnt; i_iters_fixed++)
        {
          for (int i_step_size = 0; i_step_size < sa_param_cnt; i_step_size++)
          {
            for (int i_k = 0; i_k < sa_param_cnt; i_k++)
            {
              gsl_siman_params_t params = {sa_n_tries[i_n_tries], sa_iters_fixed_t[i_iters_fixed], sa_step_size[i_step_size], sa_k[i_k], T_INITIAL, MU_T, T_MIN};
              printf("n_tries = %d, iters_fixed = %d, step_size = %.2f, k = %.2f, t_initial = %.2f, mu_t = %.2f, t_min = %.2f\n",
                     sa_n_tries[i_n_tries], sa_iters_fixed_t[i_iters_fixed], sa_step_size[i_step_size],
                     sa_k[i_k], T_INITIAL, MU_T, T_MIN);
              // printf("Random Integer: %d\n", randomIntegerInRange(1, 10000));
              // printf("Random Double: %.2f\n", randomDoubleInRange(1.0, 10000.0));

              // 3. Call the genetic_algorithm_run function
              // h = single_siman_run_param(seed, 0, &env, NULL, params); // todo
              // double *best_solution = genetic_algorithm_run(seed, &env);
              double *best_solution = tabu_search_run(seed, &env, NULL);
              // double *best_solution = pso_run(seed, &env);

              // 4. Handle or use the result
              printf("Best solution found:\n");
              exit(0);
              return;
            }
          }
        }
      }

      // random search...
      // for (int idx = 0; idx < 100; idx++)
      // {
      //   int rand_n_tries = randomIntegerInRange(1, 1000);
      //   int rand_iters_fiexed_t = randomIntegerInRange(1, 1000);
      //   double rand_step_size = randomDoubleInRange(1.0, 10000000.0);
      //   double rand_k = randomDoubleInRange(1.0, 10.0);

      //   gsl_siman_params_t params = {rand_n_tries, rand_iters_fiexed_t, rand_step_size, rand_k, T_INITIAL, MU_T, T_MIN};
      //   h = single_siman_run_param(seed, 0, &env, NULL, params); // todo
      //   printf("n_tries = %d, iters_fixed = %d, step_size = %.2f, k = %.2f, t_initial = %.2f, mu_t = %.3f, t_min = %.2f\n",
      //          rand_n_tries, rand_iters_fiexed_t, rand_step_size,
      //          rand_k, T_INITIAL, MU_T, T_MIN);
      // }

      printf("Objective = %0.3f\n", h[0]);
      printf("Hyperplane: ");
      for (int i = 1; i <= env.samples->dimension + 1; i++)
        printf("%0.5f%s", h[i], (i == env.samples->dimension + 1) ? "\n" : " ");
      // printf("Objective value: %0.3f\n", h[0]);
      printf("The number is: %d\n", k);

      reaches[k] = reach(h, env.samples);
      precisions[k] = precision(h, env.samples);
      printf("P = %0.3f\n", precisions[k]);
      if (isnan(precisions[k]))
        precisions[k] = 0;

      printf("Training: %u\t%lg\n",
             reaches[k],
             precisions[k]);

      exit(0);

      k++;
      free(h);
    }
    free(delete_samples(samples));
  }

  free(solution);
  free(delete_samples(samples_validation));
  delete_clusters_info(clusters);
  delete_clusters_info(clusters + 1);
  free(env.params);

  printf("Reaches: ");
  for (int i = 0; i < ntests; i++)
    printf("%d, ", reaches[i]);
  printf("\n");

  printf("Precisions: ");
  for (int i = 0; i < ntests; i++)
    printf("%0.3f, ", precisions[i]);
  printf("\n");

  printf("n = %d. Cluster size = %0.3f\n", n, ((double)n) / 10);

  int tot_reach = 0;
  double tot_prec = 0;
  for (int i = 0; i < ntests; i++)
  {
    tot_reach += reaches[i];
    tot_prec += precisions[i];
  }
  double avg_reach = ((double)tot_reach) / ntests;
  double avg_prec = tot_prec / ntests;

  // printf("Average reach = %0.3f, average precision = %0.3f\n", avg_reach, avg_prec);
  printf("Average reach/cluster size = %0.3f, average precision = %0.3f\n", avg_reach / (((double)n) / 20), avg_prec);
  free(reaches);
  free(precisions);

  exp_res_t res;
  res.prec = avg_prec;
  res.reach = avg_reach;
  return res;
}

int main(int argc, char *argv[])
{
  exp_res_t res;

  if (argc == 1)
    res = experiment(0);
  else
    res = experiment(atoi(argv[1]));

  printf("RESULTS: %0.5f, %0.5f\n", res.prec, res.reach);

  return 0;
}


