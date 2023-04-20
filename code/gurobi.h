/* --------------------------- Execution -------------------------------- */

/** Return a Gurobi instance intialized from the given environment */
GRBmodel *gurobi_milp(
  /** Initialization error code (see Gurobi) */
  int *state, 
  /** Instance environment */
  const env_t *);


/** Run gurobi relaxation only

 @return a new array in which 
 the zeroth element is the objective value in the relaxation solution at the end of 
 the run,
 and the next elements are the values of the decision variables. */
double *gurobi_relax(
  /** Seed for the random number generator, or NULL if the drand48 does not
   * need to be reseeded */
  unsigned int *seed, 
  /** Time limit in milliseconds */
  int tm_lim, 
  /** Time limit in milliseconds for tuning the model */
  int tm_lim_tune, 
  env_t *);

typedef struct gurobi_param {
  int threads;
  int MIPFocus;
  double ImproveStartGap;
  double ImproveStartTime;
  int VarBranch;
} gurobi_param;

/** Launch a single experiment 
 
 @return a new array in which 
 the zeroth element is the objective value in the MIP solution at the end of 
 the run,
 and the next elements are the values of the decision variables. */
double *single_gurobi_run(
  /** Seed for the random number generator, or NULL if the drand48 does not
   * need to be reseeded */
  unsigned int *seed, 
  /** Time limit in milliseconds */
  int tm_lim, 
  /** Time limit in milliseconds for tuning the model */
  int tm_lim_tune, 
  env_t *,
  double *,
  gurobi_param *);
