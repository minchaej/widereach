#include <math.h>

#include "widereach.h"

double single_run(unsigned int *seed, int tm_lim, env_t *env) {
    samples_t *samples = env->samples;
    env->solution_data = solution_data_init(samples_total(samples));
        
    if (seed !=NULL) {
        srand48(*seed);
    }

    glp_prob *p = milp(env);
    // glp_write_lp(p, NULL, "tmp.lp");
    glp_scale_prob(p, GLP_SF_AUTO);
    glp_simplex(p, NULL);

    glp_iocp *parm = iocp(env);
    parm->tm_lim = tm_lim;
    parm->bt_tech = GLP_BT_DFS;
    // parm->bt_tech = GLP_BT_BLB;
    /* MFV chooses the largest {x} (e.g., 0.99 in favor of 0.1)
    * It would be similar to branch_target=1 for the positive samples,
    * but the opposite for negative samples */
    // parm->br_tech = GLP_BR_LFV;
    glp_intopt(p, parm);
    free(parm);

    double obj = glp_mip_obj_val(p);
    glp_printf("Objective: %g\n", obj);
    /*
    int index_max = violation_idx(0, env.samples);
    for (int i = 1; i <= index_max; i++) {
        glp_printf("%s:\t%g\n", glp_get_col_name(p, i), glp_mip_col_val(p, i));
    }*/

    glp_delete_prob(p);
    free(delete_solution_data(env->solution_data));
    
    return obj;
}

#define TM_LIM_SEARCH 5000
#define SCALE_SEARCH 20

double ticks2threshold(unsigned int ticks) {
  return ticks / (double) SCALE_SEARCH;
}

unsigned int threshold2ticks(double threshold) {
  return (unsigned int) round(threshold * SCALE_SEARCH);
}

void advance_search(
    unsigned int *middle, 
    unsigned int *moving, 
    unsigned int hinge) {
  *moving = *middle;
  *middle = (*middle + hinge) / 2;
}

double precision_threshold(unsigned int *seed, env_t *env) {
  params_t *parms = env->params;
  
  unsigned int left = 1;
  unsigned int right = SCALE_SEARCH;
  unsigned int middle = threshold2ticks(parms->theta);
  
  double obj;
  do {
    parms->theta = ticks2threshold(middle);
    glp_printf("Search, theta = %g\n", parms->theta);
    obj = single_run(seed, TM_LIM_SEARCH, env);
    if (obj > 0.) {
      advance_search(&middle, &left, right);
    } else if (obj < 0.) {
      advance_search(&middle, &right, left);
    }
  } while (left < right - 1 && obj != 0.);
  
  return parms->theta;
}
  
