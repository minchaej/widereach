#include "widereach.h"
#include "helper.h"

#include <pthread.h>
#include <unistd.h>

double *best_sol;
double *rel_sol;
sys7_t *rel_sys;
double best_obj;
pthread_t thread;
pthread_mutex_t lock;
int running = 0;
int soln_by_hps = 0; // = 1 if best solution was found by hyperplanes

void *generate_hyperplanes(void *vparg) {
  //cb_arg *arg = vparg;
  env_t *env = vparg;
  //GRBmodel *model = arg->model;

  int state, nvars;
  /*printf("About to get nvars\n");
  state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  printf("Got nvars = %d\n", nvars);*/

  int idx_max = violation_idx(0, env->samples);

  double *h, *soln, obj;
  while(running) {
    //h = best_random_hyperplane(1, env);
    //h = random_hyperplane(env->samples->dimension);
    if(!rel_sys) {
      h = random_hyperplane(env->samples->dimension);
    } else {
      /*if(rel_mat->size1 > 1000 || rel_mat->size2 > 1000) {
	printf("Fixing corruption\n");
	rel_mat = generate_fixedpt_mat(env, rel_sol); //TEMP FIX - DATA CORRUPTED
	}*/
      pthread_mutex_lock(&lock);
      gsl_matrix *A = gsl_matrix_alloc(rel_sys->A->size1, rel_sys->A->size2);
      gsl_matrix_memcpy(A, rel_sys->A);
      gsl_vector *b = gsl_vector_alloc(rel_sys->b->size);
      gsl_vector_memcpy(b, rel_sys->b);
      pthread_mutex_unlock(&lock);
      sys7_t *s = malloc(sizeof(sys7_t));
      s->A = A;
      s->b = b;
      h = random_constrained_hyperplane(env, rel_sol, s);
      gsl_matrix_free(A);
      gsl_vector_free(b);
      free(s);
    }
    soln = blank_solution(env->samples);
    obj = hyperplane_to_solution(h, soln, env);
    if(obj > best_obj) {
      pthread_mutex_lock(&lock);
      soln_by_hps = 1;
      
      best_obj = obj;
      //printf("Hyperplane solution:\n");
      for(int i = 0; i <= env->samples->dimension+1; i++)
	best_sol[i] = GRB_UNDEFINED;
      for(int i = env->samples->dimension+1; i < idx_max; i++) {
	best_sol[i] = soln[i+1]; //TODO: indices
	//printf("%0.3f ", best_sol[i]);
      }

      pthread_mutex_unlock(&lock);
    }
    free(h);
    free(soln);
  }

  return NULL;
}

int backgroundHyperplanes(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  int state = 0;
  env_t *env = usrdata;
  if(where == GRB_CB_MIPSOL) {
    int nvars;
    state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
    
    double obj;
    state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_OBJ, &obj);
    if(obj > best_obj) {
      double *soln = CALLOC(nvars, double);
      state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, soln);
      pthread_mutex_lock(&lock);
      soln_by_hps = 0;
      
      best_obj = obj;
      /*for(int i = 0; i < nvars; i++) {
	best_sol[i] = soln[i];
	}*/
      
      pthread_mutex_unlock(&lock);
      free(soln);
    }
  } else if(where == GRB_CB_MIPNODE) {
    double obj;
    state |= GRBcbget(cbdata, where, GRB_CB_MIPNODE_OBJBST, &obj);
    if ((obj < best_obj) && soln_by_hps) {
      //random hyperplanes have found a better solution
      double newobj;
      pthread_mutex_lock(&lock);
      state |= GRBcbsolution(cbdata, best_sol, &newobj);
      printf("submitted new soln with objective %g\n", newobj);
      pthread_mutex_unlock(&lock);
    }
    int rel_state = GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, rel_sol);
    if(rel_state == 10005) {
      return state; //sometimes this error happens for some reason - gurobi bug? just skip it. usually doesn't happen
    } else {
      state |= rel_state;
    }



    
    if(rand() <= RAND_MAX) {
      pthread_mutex_lock(&lock);
      if(rel_sys) {
	gsl_matrix_free(rel_sys->A);
	gsl_vector_free(rel_sys->b);
	free(rel_sys);
      }
      //this section is error-checking - can be removed
      /*int nvars;
      state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
      for(int i = env->samples->dimension; i < nvars; i++) {
	if(rel_sol[i] > 1) {
	  printf("i = %d, rel_sol[i] = %g\n", i, rel_sol[i]);
	  exit(0);
	}
	}*/
      rel_sys = generate_fixedpt_mat(env, rel_sol);
      pthread_mutex_unlock(&lock);
    }
    //sleep(100000000);
  }
  return state;
}

void startHyperplanes(env_t *env, GRBmodel *model) {
  running = 1;
  best_sol = blank_solution(env->samples);
  rel_sol = blank_solution(env->samples);
  pthread_mutex_init(&lock, NULL);
  pthread_create(&thread, NULL, generate_hyperplanes, env);
  //pthread_join(thread, NULL);
}

void stopHyperplanes() {
  running = 0;
  pthread_join(thread, NULL);
  pthread_mutex_destroy(&lock);
  free(best_sol);
  free(rel_sol);
}
