#include "widereach.h"
#include "helper.h"

#define TRY_MODEL(premise, step) \
  TRY(premise, \
      error_handler(state, model, step), \
      NULL)

#define MSG_LEN 256

int error_handler(int state, GRBmodel *model, char *step) {
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

void *compute_inseparabilities(env_t *env, int method) { //this is void* for now because I don't know what to return (but should be ptr to work with TRY)
  int state;
  GRBmodel *model;
  
  TRY_MODEL(model = gurobi_milp(&state, env), "model creation");
  //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "LogToConsole", 0), "disable console output");

  TRY_MODEL(state = GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"), "set log file");

  TRY_MODEL(state = GRBupdatemodel(model), "update model");
  
  //fix all integer decision vars
  int nvars;
  TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars), "get number of variables");

  int i;
  
  for(i = 0; i < env->samples->count[1]; i++) {
    TRY_MODEL(state = GRBsetdblattrelement(model, "LB", env->samples->dimension + 1 + i, 1), "set LB");
    TRY_MODEL(state = GRBsetdblattrelement(model, "UB", env->samples->dimension + 1 + i, 1), "set UB");
  }

  for(; i < env->samples->count[0] + env->samples->count[1]; i++) {
    TRY_MODEL(state = GRBsetdblattrelement(model, "LB", env->samples->dimension + 1 + i, 0), "set LB");
    TRY_MODEL(state = GRBsetdblattrelement(model, "UB", env->samples->dimension + 1 + i, 0), "set UB");
  }

  //TRY_MODEL(state = GRBcomputeIIS(model), "compute IIS");
  //TRY_MODEL(state = GRBwrite(model, "tmp.ilp"), "write iis");
    /*int *out = CALLOC(1, int);
  int ninfeas = *out;

  int nconstrs;
  TRY_MODEL(state = GRBgetintattr(model, "NumConstrs", &nconstrs), "get number of constraints");
  for(int j = 0; j < nconstrs; j++) {
    int iniis;
    TRY_MODEL(state = GRBgetintattrelement(model, "IISConstr", j, &iniis), "check if in iis");
    if(iniis) ninfeas++;
  }

  return out;*/



    int nconstrs;
  TRY_MODEL(state = GRBgetintattr(model, "NumConstrs", &nconstrs), "get number of constraints");
  double *rhspen = malloc(nconstrs*sizeof(double));
  for(int i = 0; i < nconstrs; i++) {
    rhspen[i] = 1;
  }

  double *lbpen = malloc(nvars*sizeof(double));
  double *ubpen = malloc(nvars*sizeof(double));
  for(int i = 0; i < nvars; i++) {
    lbpen[i] = GRB_INFINITY;
    ubpen[i] = GRB_INFINITY;
  }

    
  TRY_MODEL(state = GRBfeasrelax(model, method, 0, lbpen, ubpen, rhspen, NULL), "compute feasibility relaxation");
  free(rhspen);
  free(lbpen);
  free(ubpen);
  TRY_MODEL(state = GRBwrite(model, "feasrelax.lp"), "write feas relax");
  
  TRY_MODEL(state = GRBoptimize(model), "optimize feasibility relaxation");
  
  double *total_viol = malloc(sizeof(double));

  TRY_MODEL(state = GRBgetdblattr(model, "ObjVal", total_viol), "get feas relax objective");

  double *x = malloc(nvars*sizeof(double));
  TRY_MODEL(state = GRBgetdblattrarray(model, "X", 0, nvars, x), "get feas relax vars");
  /*
printf("Result:\n");

  for(int i = 0; i <= env->samples->dimension; i++) {
    //printf("%g%s", x[i], i == nvars-1 ? "\n" : " ");
    printf("x[%d] = %g\n", i, x[i]);
    }*/
  
  free(x);

  return total_viol;
}
