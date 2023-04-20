#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <time.h>

#define TRY_MODEL(premise, step) \
  TRY(premise, \
      error_handle(state, model, step), \
      NULL)
  
#define MSG_LEN 256

#define FREQ 1000
#define LIM 500
  
int error_handle(int state, GRBmodel *model, char *step) {
  if (!state) {
    return 0;
  }
  
  GRBenv *env = GRBgetenv(model);
  char msg[MSG_LEN];
  snprintf(msg, MSG_LEN, 
           "Error (%s): %i\nError message: %s\n", 
           step, state, GRBgeterrormsg(env));
  GRBmsg(env, msg);
  return state;
}

double *gurobi_relax(unsigned int *seed, int tm_lim, int tm_lim_tune, env_t *env) {
    /* samples_t *samples = env->samples;
    env->solution_data = solution_data_init(samples_total(samples));
    
    if (seed != NULL) {
        srand48(*seed);
    }*/

    int state;
    GRBmodel *model;
    
    TRY_MODEL(model = gurobi_milp(&state, env), "model creation");

    TRY_MODEL(state = GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"), "set log file");

    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TuneTimeLimit", 
                             tm_lim_tune / 1000.),
      "set time limit for tuning")
    
    printf("optimize ...\n");

    double k;
    TRY_MODEL(GRBupdatemodel(model), "update model");
    TRY_MODEL(GRBgetdblattr(model, "Kappa", &k), "get condition number");
    printf("Kappa = %.3e\n", k);
    
    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TimeLimit", 
                             tm_lim / 1000.),
      "set time limit");

    /*GRBwrite(model, "pre.prm"); //save params before tuning

    TRY_MODEL(state = GRBtunemodel(model), "autotune");
    int nresults;
    TRY_MODEL(state = GRBgetintattr(model, "TuneResultCount", &nresults), "get tune results");
    printf("------------%d results--------------\n", nresults);
    if(nresults > 0)
      TRY_MODEL(state = GRBgettuneresult(model, 0), "apply tuning");
      GRBwrite(model, "post.prm");
    
      GRBwrite(model, "tmp.lp");*/

    //convert integer vars to double
    int nvars;
    TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars), "get number of variables");
    char *conts = CALLOC(nvars, char);
    char *vtypes = CALLOC(nvars, char);
    TRY_MODEL(state = GRBgetcharattrarray(model, "VType", 0, nvars, vtypes), "get variable types");
    for(int i = 0; i < nvars; i++)
      conts[i] = GRB_CONTINUOUS;
    TRY_MODEL(state = GRBsetcharattrarray(model, "VType", 0, nvars, conts), "set vars to continuous");

    free(conts);

    TRY_MODEL(state = GRBoptimize(model), "optimize");

    int optimstatus;
    TRY_MODEL(
      state = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus), 
      "get optimization status")

    //find optimal value of decision vars
    double *result = CALLOC(nvars+1, double);
    TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result), "get objective value");
    TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result+1), "get decision vars");

    /*printf("Decision vars:\n");
    for(int i = 0; i < nvars+1; i++)
    printf("%0.3f%s", result[i], (i == nvars) ? "\n" : ", ");*/

    //GRBwrite(model, "soln.json");

    TRY_MODEL(state = GRBsetcharattrarray(model, "VType", 0, nvars, vtypes), "reset vars to discrete");
    free(vtypes);
    return result;
}
int ncalls = 0;
int testCB(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPNODE) return 0;
  if(ncalls++ % 5000 != 0) return 0;
  int nvars;
  int state;
  state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  double *soln = CALLOC(nvars, double);
  state = GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, soln);
  /*printf("STATE = %d\n", state);
  printf("ERROR: %s STOP\n", GRBgeterrormsg(GRBgetenv(model)));
  printf("------RELAXATION SOLUTION: %0.3f %0.3f %0.3f-------\n", soln[0], soln[1], soln[2]);*/
  printf("----------------------------\nRelaxation Solution:\n");
  for(int i = 0; i < nvars; i++)
    printf("%0.3f ", soln[i]);
  printf("\n----------------------------\n");
  free(soln);
  return 0;
}

int test_heur(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPNODE) return 0;
  int nvars;
  int state;
  env_t *env = usrdata;
  state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  double *h = CALLOC(nvars, double); //relaxation solution
  state = GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, h);
  double *solution = blank_solution(env->samples);
  double obj = hyperplane_to_solution(h, solution, env);
  state = GRBcbsolution(cbdata, solution, NULL);
  int r = reach(solution, env->samples);
  double p = precision(solution, env->samples);
  printf("[HEUR] Solution with obj = %0.3f. Reach = %d, prec = %0.3f\n", obj, r, p);
  /*printf("       Hyperplane: ");
  for(int i = 0; i < env->samples->dimension+1; i++) printf("%0.3f ", h[i]);
  printf("\n");*/
  free(h);
  free(solution);
  return 0;
}

int modified = 0;
int k = 0;
int modified_branching(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPSOL) return 0;
  if(k++ < 10) return 0;
  int state;
  GRBterminate(model);
  //state = GRBsetintattrelement(model, GRB_INT_ATTR_BRANCHPRIORITY, 30, 1);
  modified = 1;
  k = 0;
  //printf("state = %d\n", state);
  return state;
}

int gurobi_random_bounded(int *branched, int idx_min, int idx_max) {
  int *eligible = CALLOC(idx_max - idx_min + 1, int);
  int eligible_cnt = 0;
  for (int i = idx_min; i <= idx_max; i++) {
    if (!branched[i]) {
      eligible[eligible_cnt++] = i;
    }
  }
  int candidate = eligible_cnt > 0 ? eligible[lrand48() % eligible_cnt] : -1;
  free(eligible);
  return candidate;
}

typedef struct pair_t {
  int key;
  double val;
} pair_t;

int cmppairs(const void *a, const void *b) {
  return ((pair_t *) a)->val > ((pair_t *) b)->val;
}

/* Returns an array containing the num closest samples to the hyperplane
 * This can be done faster (linear time) using order statistics */
int *gurobi_by_violation(env_t *env, double *sol, int num) {
  int *cands = CALLOC(num, int);
  double branch_target = env->params->branch_target;
  double max_frac = -__DBL_MAX__;
  int max_idx = -1;
  samples_t *samples = env->samples;
  int idx_max = violation_idx(0, samples);
  pair_t *pairs = CALLOC(idx_max, pair_t);
  for (int i = samples->dimension+2; i < idx_max; i++) {
    //double value = glp_get_col_prim(p, i);
    double value = sol[i];
    if (index_label(i, samples) > 0) {
      value = 1. - value;
    }
    value = fabs(value - branch_target);
    pairs[i] = (pair_t) {i, value};
  }
  qsort(pairs, idx_max, sizeof(pair_t), cmppairs);
  for(int i = 0; i < num; i++)
    cands[i] = pairs[i].key;
  free(pairs);
  return cands;
}

double *incumb;
int modified_rins(GRBmodel *parent_model, void *cbdata, int where, void *usrdata) {
  env_t *env = usrdata;
  if(where == GRB_CB_MIPSOL) {
    int nvars;
    int state;
    state = GRBgetintattr(parent_model, GRB_INT_ATTR_NUMVARS, &nvars);
    if(k++ == 0)
      incumb = CALLOC(1, double);
    free(incumb);
    incumb = CALLOC(nvars, double);
    state = GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, incumb);
    //env->solution_data->integer_solution = CALLOC(nvars, double);
    //for(int i = 0; i < nvars; i++)
    //  env->solution_data->integer_solution[i] = incumb[i];
    /*for(int i = env->samples->dimension + 2; i < nvars; i++) {
      printf("incumb[%d] = %0.3f\n", i, incumb[i]);
      }*/
    //free(incumb);
    return state;
  } if(where == GRB_CB_MIPNODE && k++ % FREQ == 0) {
    int state = 0;
    
    GRBmodel *model = GRBcopymodel(parent_model);
    //GRBsetcallbackfunc(model, test_heur, usrdata);
    GRBsetcallbackfunc(model, NULL, NULL);
    state |= GRBsetintparam(GRBgetenv(model), "RINS", 0);
    GRBupdatemodel(model);
  
    int nvars;
    state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
    printf("Got nvars. State = %d\n", state);
    double *h = CALLOC(nvars, double); //relaxation solution
    state |= GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, h);
    //printf("Got relaxation. State = %d\n", state);
    if(state == 10005) return 0; //sometimes cannot load relaxation
    double *solution = blank_solution(env->samples);
    hyperplane_to_solution(h, solution, env);
    //double *incumb = env->solution_data->integer_solution;
    /*for(int i = env->samples->dimension + 2; i < nvars; i++) {
      if(incumb[i] != 0 && incumb[i] != 1)
	printf("[HEUR] NON-INTEGER-INCUMBENT: incumb[%d] = %0.3f\n", i, incumb[i]);
	}*/
    int common = 0;
    for(int i = env->samples->dimension + 2; i < nvars; i++) {
      if(incumb[i] == solution[i]) {
	//fix value by setting LB and UB to the same value
	state |= GRBsetdblattrelement(model, "LB", i, incumb[i]);
	state |= GRBsetdblattrelement(model, "UB", i, incumb[i]);
	common++;
      }
    }
    printf("Fixed vars. State = %d\n", state);

    if(common == 0) {
      printf("[HEUR] subproblem identical to original\n");
      return state;
    }

    state |= GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0, nvars, incumb);
    printf("Set start. State = %d\n", state);
    /*printf("Incumbent:\n");
    for(int i =0; i < nvars; i++) {
      printf("%0.3f%s", incumb[i], (i == nvars - 1) ? "\n" : " ");
      }*/


    state |= GRBsetdblparam(GRBgetenv(model), "NodeLimit", LIM);
    //state |= GRBsetintparam(GRBgetenv(model), "OutputFlag", 0);
    printf("State = %d\n", state);
    printf("[HEUR] STARTING SUBPROBLEM (fixed %d vars):\n", common);

    state |= GRBoptimize(model);
    printf("State = %d\n", state);

    printf("[HEUR] END SUBPROBLEM\n");

    double *new_sol = CALLOC(nvars, double);
    state |= GRBgetdblattrarray(model, "X", 0, nvars, new_sol);
    printf("State = %d\n", state);

    state |= GRBcbsolution(cbdata, new_sol, NULL);
    //  printf("[HEUR] STATE = %d\n", state);
    free(h);
    free(solution);
    //free(incumb);
    free(new_sol);
    GRBfreemodel(model);
    return state;
  }
  return 0;
}

double *single_gurobi_run(unsigned int *seed, 
                          int tm_lim, 
                          int tm_lim_tune, 
                          env_t *env,
			  double *warm_start,
			  gurobi_param *param_setting) {
    samples_t *samples = env->samples;
    env->solution_data = solution_data_init(samples_total(samples));
    
    if (seed != NULL) {
      printf("Applying seed %u\n", *seed);
        srand48(*seed);
    }

    int state;
    GRBmodel *model;

    ncalls = 0;
    
    TRY_MODEL(model = gurobi_milp(&state, env), "model creation");

    TRY_MODEL(state = GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"), "set log file");

    TRY_MODEL(state = GRBupdatemodel(model), "update model");
    
    int nvars;
    TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars), "get number of variables");
    
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "ScaleFlag", 3), "set scale flag");
    

    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TuneTimeLimit", 
                             tm_lim_tune / 1000.),
      "set time limit for tuning")
      //TRY_MODEL(state = GRBtunemodel(model), "parameter tuning")
    
    
    // Cut generation (TODO warning: just goofing around)
    /* GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0);
    GRBsetintparam(GRBgetenv(model), "CoverCuts", 2);
    GRBsetintparam(GRBgetenv(model), "ImpliedCuts", 2);
    GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2); */
    
    printf("optimize ...\n");

    double k;
    TRY_MODEL(GRBupdatemodel(model), "update model");
    TRY_MODEL(GRBgetdblattr(model, "Kappa", &k), "get condition number");
    printf("Kappa = %.3e\n", k);
    
    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TimeLimit", 
                             tm_lim / 1000.),
      "set time limit")

      /*GRBwrite(model, "pre.prm"); //save params before tuning

    TRY_MODEL(state = GRBtunemodel(model), "autotune");
    int nresults;
    TRY_MODEL(state = GRBgetintattr(model, "TuneResultCount", &nresults), "get tune results");
    printf("------------%d results--------------\n", nresults);
    if(nresults > 0)
      TRY_MODEL(state = GRBgettuneresult(model, 0), "apply tuning");
      GRBwrite(model, "post.prm");*/
    
    GRBwrite(model, "tmp.lp");

    
    if(warm_start) {
      TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Method", 0), "use only simplex");
      TRY_MODEL(state = GRBsetdblattrarray(model, "Start", 0, nvars, warm_start), "apply relaxation solution");
      TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Presolve", 0), "disable presolve");
    }

    if(1) {
    printf("Generating best of %d hyperplanes\n", env->params->rnd_trials);
    double *h = best_random_hyperplane(1, env);
    printf("Dimension = %lu\n", env->samples->dimension);
    for(int i = 0; i < env->samples->dimension+1; i++) h[i] /= 100;
    //printf("Hyperplane: %0.3f %0.3f %0.3f %0.3f\n", h[0], h[1], h[2], h[3]);
    printf("Hyperplane: ");
    for(int i = 0; i < env->samples->dimension+1; i++)
      printf("%0.3f%s", h[i], (i == env->samples->dimension) ? "\n" : " ");
    printf("Done, printing solution:\n");
    double *random_solution = blank_solution(samples);
    double random_objective_value = hyperplane_to_solution(h, random_solution, env);
    printf("Objective value = %0.3f\n", random_objective_value);

    /*int idx_max = violation_idx(0, samples);
    for(int i = 0; i < idx_max; i++) {
      //printf("var %d = %0.3f%s\n", i, random_solution[i], (i == idx_max) ? "\n" : " ");
      char *name = CALLOC(100, char);
      TRY_MODEL(state = GRBgetstrattrelement(model, "VarName", i, &name), "get var");
      printf("%s = %0.3f\n", name, random_solution[i]);
      }*/

    TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0, nvars, random_solution + 1), "set start");

    free(h);
    free(random_solution);
    }
    

    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CliqueCuts", 2), "set cuts");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "FlowCoverCuts", 2), "set cuts");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2), "set cuts");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CoverCuts", 2), "set cuts");
    //GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0);

    //TRY_MODEL(state = GRBsetcallbackfunc(model, testCB, NULL), "add callback");
    //TRY_MODEL(state = GRBsetcallbackfunc(model, test_heur, env), "add custom heuristic");
    //TRY_MODEL(state = GRBsetcallbackfunc(model, modified_rins, env), "add modified RINS heuristic");

    //TRY_MODEL(state = GRBsetcallbackfunc(model, modified_branching, env), "modify branching strategy");
    //variable hints:
    if(0) {
    double *hint = CALLOC(nvars, double);
    int i;
    for(i = 0; i <= env->samples->dimension; i++)
      hint[i] = GRB_UNDEFINED;
    for(; i <= env->samples->dimension + env->samples->count[1]; i++)
      hint[i] = 1;
    for(; i <= env->samples->dimension + env->samples->count[1] + env->samples->count[0]; i++)
      hint[i] = 0;
    hint[i] = GRB_UNDEFINED;
    }
    
    if(param_setting) {
      TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Threads", param_setting->threads), "set thread limit");
      TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "MIPFocus", param_setting->MIPFocus), "set focus");
      TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "ImproveStartGap", param_setting->ImproveStartGap), "set start gap");
      TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "ImproveStartTime", param_setting->ImproveStartTime), "set start time");
      TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "VarBranch", param_setting->VarBranch), "set branching strategy");
      //TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "Heuristics", *param_setting/((double) 10)), "set heuristics");
      //TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "Heuristics", 0.1), "set heuristics");
      /*switch(*param_setting) {
      case 1:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CliqueCuts", 2), "set cuts");
	break;
      case 2:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 3:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "FlowCoverCuts", 2), "set cuts");
	break;
      case 4:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "FlowCoverCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 5:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2), "set cuts");
	break;
      case 6:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 7:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CoverCuts", 2), "set cuts");
	break;
      case 8:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CoverCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 9:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "CoverCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 10:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "FlowCoverCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      case 11:
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "InfProofCuts", 1), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "FlowCoverCuts", 2), "set cuts");
	TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0), "disable gomory");
	break;
      default:
	break;
	}*/
    }

    //TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "NodeLimit", 0), "set node limit");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Threads", 1), "set thread limit");
    //TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "Heuristics", 0), "disable heuristics");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Cuts", 0), "disable cutting planes");
    //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "RINS", 0), "disable RINS");

    int optimstatus;
    int dimension = env->samples->dimension;
    int *branched = CALLOC(nvars, int);

    do {
      if(modified) {
	//choose branching var
	/*int candidate = gurobi_random_bounded(branched, dimension + 1, dimension + samples_total(samples));
	GRBsetintattrelement(model, GRB_INT_ATTR_BRANCHPRIORITY, candidate, 1);
	branched[candidate] = 1;
	printf("Candidate = %d\n", candidate);*/
	double *sol = CALLOC(nvars, double);
	TRY_MODEL(state = GRBgetdblattrarray(model, "X", 0, nvars, sol), "get intermediate X");
	int num_cands = 50;
	int *cands = gurobi_by_violation(env, sol, num_cands);
	for(int i = 0; i < num_cands; i++) {
	  TRY_MODEL(GRBsetintattrelement(model, GRB_INT_ATTR_BRANCHPRIORITY, cands[i], 1), "set branch priority");
	}
	free(cands);
      }
      modified = 0;
      TRY_MODEL(state = GRBoptimize(model), "optimize");
      TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus), "get optimization status");
      if(optimstatus == GRB_OPTIMAL) break;
    } while(modified);

    //find optimal value of decision vars
    printf("%d variables\n", nvars);
    
    //NOTE: modified - should be nvars and not include the objval
    double *result = CALLOC(nvars+1, double);
    TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result), "get objective value");
    TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result+1), "get decision vars");

    /*double *result = CALLOC(nvars, double);
      TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result), "get decision vars");*/

    /*printf("Decision vars:\n");
    for(int i = 0; i < nvars; i++)
    printf("%0.3f%s", result[i], (i == nvars - 1) ? "\n" : " ");*/

    GRBwrite(model, "soln.sol");

    //printf("--------------------------%d calls--------------------------\n", ncalls);

    return result;
	      

    /*
    // glp_scale_prob(p, GLP_SF_AUTO);
    glp_simplex(p, NULL);

    glp_iocp *parm = iocp(env);
    parm->tm_lim = tm_lim;
    parm->bt_tech = GLP_BT_DFS;
    // parm->bt_tech = GLP_BT_BLB;
    MFV chooses the largest {x} (e.g., 0.99 in favor of 0.1)
    * It would be similar to branch_target=1 for the positive samples,
    * but the opposite for negative samples 
    // parm->br_tech = GLP_BR_LFV;
    glp_intopt(p, parm);
    free(parm);

    double *result = solution_values_mip(p);
     size_t dimension = samples->dimension;
    double *result = CALLOC(dimension + 2, double);
    double *h = hyperplane(p, samples);
    copy_hyperplane(dimension, result, h);
    free(h); 
    double obj = result[0] = glp_mip_obj_val(p);
    glp_printf("Objective: %g\n", obj);
    // result[dimension + 1] = obj;
    
    int index_max = violation_idx(0, env.samples);
    for (int i = 1; i <= index_max; i++) {
        glp_printf("%s:\t%g\n", glp_get_col_name(p, i), glp_mip_col_val(p, i));
    }

    glp_delete_prob(p);
    free(delete_solution_data(env->solution_data));
    
    // return result; */
    return NULL;
}
