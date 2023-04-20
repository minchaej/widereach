#include <stdio.h>

#include "widereach.h"

#include "helper.h"

#include <math.h>

#define LINE_MAX 255

#define UNUSED_RESULT(cmd) (void) (cmd + 1)

void compare_files(char *filename) {
    char line[LINE_MAX];
    snprintf(line, LINE_MAX, "cmp %s tmp.lp", filename);
	FILE *cmp = popen(line, "r");
	printf("%s\n", fgets(line, sizeof(line), cmp) == NULL ? 
			"success" : "FAILURE");
	pclose(cmp);
	UNUSED_RESULT(system("rm tmp.lp"));
}

/*
void normalize_samples(env_t *env) {
  size_t dim = env->samples->dimension;
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    double *means = CALLOC(dim, double);
    double *stddevs = CALLOC(dim, double);
    for(int d = 0; d < dim; d++) {
      //selected class c, feature d
      double sum = 0;
      for(int i = 0; i < cnt; i++)
	sum += env->samples->samples[c][i][d];
      means[d] = sum/cnt;
      double sds = 0; //sum of (xi - mu)^2
      for(int i = 0; i < cnt; i++)
	sds += pow(env->samples->samples[c][i][d]-means[d], 2);
      stddevs[d] = sqrt(sds/cnt);
      
      for(int i = 0; i < cnt; i++)
	env->samples->samples[c][i][d] = (env->samples->samples[c][i][d]-means[d])/stddevs[d];
    }
    free(means);
    free(stddevs);
  }
}

void feature_scaling(env_t *env) {
  size_t dim = env->samples->dimension;
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    for(int d = 0; d < dim; d++) {
      //selected class c, feature d
      double norm = 0;
      for(int i = 0; i < cnt; i++)
	norm += fabs(env->samples->samples[c][i][d]);

      if(norm == 0){
	printf("Found norm 0 in feature scaling\n");
	continue;
      }
                  
      for(int i = 0; i < cnt; i++)
	env->samples->samples[c][i][d] /= norm;
    }
  }  
  }*/

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

double *init_solution(int nmemb, double *solution) {
  for (int i = 1; i < nmemb; i++) {
    solution[i] = .5;
  }
  return solution;
}

int main(int argc, char *argv[]) {
	env_t env;
	env.params = params_default();
	env.params->theta = 0.99;
	int n = 400;
	env.params->lambda = 100 * (n + 1);
	srand48(20200621154912);
	//env.samples = random_samples(n, n / 2, 8);
	FILE *infile = fopen("sample.dat", "r");
	  env.samples = read_binary_samples(infile);
	clusters_info_t clusters[2];
	clusters_info_singleton(clusters, 220, 2);
	clusters_info_t *info = clusters + 1;
	info->dimension = 1;
	info->cluster_cnt = 2;
	info->count = CALLOC(2, size_t); 
	info->count[0] = 300;
	info->count[1] = 200;
	info->shift = CALLOC(2, double);
	info->side = CALLOC(2, double);
	info->shift[0] = 0.;
	info->side[0] = 1.;
	info->shift[1] = .9;
	info->side[1] = .1;
	//env.samples = random_sample_clusters(clusters);
	simplex_info_t simplex_info = {
	  .count = n,
	  .positives = n / 5,
	  .cluster_cnt = 2,
	  .dimension = 4,
	  .side = sqrt(24.0/40320.0)
	};
	//env.samples = random_simplex_samples(&simplex_info);

	//normalize_samples(&env);
	feature_scaling(&env);
	env.solution_data = solution_data_init(n);

	/*glp_prob *p = milp(&env);

	printf("Integration testing: samples\n");
	print_samples(env.samples);
	printf("\nIntegration testing: CPLEX LP compare\n");
	glp_write_lp(p, NULL, "tmp.lp");
	printf("Comparison result:\t");
    compare_files("itest.lp");

	printf("\nIntegration testing: solve relaxation\n");
	glp_simplex(p, NULL);

	printf("\nIntegration testing: solve integer problem\n");
	glp_iocp *parm = iocp(&env);
	glp_intopt(p, parm);
	free(parm);

	glp_delete_prob(p);
    
    printf("\nIntegration test: interdiction\n");
    p = init_consistency_problem(2);
    sample_locator_t loc;
    for (size_t i = 0; i < 2; i++) {
        loc.class = i;
        for (int j = 0; j < 2; j++) {
            loc.index = j;
            append_sample(p, &loc, &env);
        }
    }
    loc.class = 0;
    loc.index = 2;
    int consistency = is_interdicted(p, &loc, &env);
    glp_write_lp(p, NULL, "tmp.lp");
    printf("%s\n", !consistency ? "FAILURE": "success");
    glp_delete_prob(p);
    printf("Comparison result:\t");
    compare_files("consistency.lp");
    
    printf("\nIntegration test: obstruction\n");
    sample_locator_t target = { 1, 0 };
    sample_locator_t source, obstruction[2];
    source.class = 1;
    source.index = 1;
    sample_locator_t *obstruction_ptr[2];
    for (int i = 0; i < 2; i++) {
        obstruction[i].class = 0;
        obstruction[i].index = i;
        obstruction_ptr[i] = obstruction + i;
    }
    int status = 
        is_obstructed(&target, &source, 2, obstruction_ptr, env.samples);
	printf("%s (%i)\n", status ? "FAILURE": "success", status);
    
    printf("\nIntegration testing: Gurobi LP compare\n");
    int state;
    GRBmodel *model = gurobi_milp(&state, &env);
    GRBwrite(model, "tmp.lp");
    printf("Comparison result:\t");
    compare_files("gitest.lp");*/

	//double *result = single_gurobi_run(NULL, 120000, 3000, &env);
	//double *result = single_gurobi_run(NULL, 120000, 3000, &env, NULL, NULL);

	int solution_size = env.samples->dimension + samples_total(env.samples) + 3;
	double *solution = CALLOC(solution_size, double);
	double *h = gurobi_relax(NULL, 120000, 3000, &env);
	printf("-----------------------------FINISHED RELAXATION--------------------------------------\n");
	printf("Objective value: %0.3f\n", *h);
	hyperplane_to_solution_parts(h+1, init_solution(solution_size, solution), env.params, env.samples);
	printf("Results: %u\t%lg\n", reach(solution, env.samples), precision(solution, env.samples));
	
	//double *result = single_gurobi_run(NULL, 12000000, 300000, &env, relres+1);
	//double *result = single_gurobi_run(NULL, 12000000, 300000, &env, NULL);
	//double *result = single_run(NULL, 120000000, &env);

	free(h);
	//free(result);
	/*printf("\n\nSamples:\n");
	  print_samples(env.samples);*/

	delete_env(&env);
}


