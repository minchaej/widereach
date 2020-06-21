/*
 * Convert a problem instance (env) into GLPK problem (glp_prob).
 *
 * Numbering Conventions
 *
 * Variables (columns)
 * 1 to dimension	w
 * dimension+1	c
 * dimension+2 to dimension+positives+1 xi
 * dimension+positives+2 to dimension+samples+1	yj
 * dimension+samples+2	V
 *
 * Constrains (rows)
 * 1 to positives	xi
 * positives+1 to samples	yj
 * samples+1	V 
 */

#include <stdio.h>

#include "widereach.h"
#include "helper.h"

#define NAME_LEN_MAX 255

glp_iocp *iocp(const params_t *params)  {
	glp_iocp *parm = CALLOC(1, glp_iocp);
	glp_init_iocp(parm);
	parm->msg_lev = params->verbosity;
	parm->pp_tech = GLP_PP_NONE;
	parm->sr_heur = GLP_OFF;
	parm->binarize= GLP_ON;
	// parms->cb_func = callback;
	// parms->cb_info =;
	// parms->cb_size =;
	return parm;
}


glp_prob *init_prob(const env_t *env) {
	glp_prob *p = glp_create_prob();
	params_t *params = env->params;
	glp_set_prob_name(p, params->name);
	glp_set_obj_dir(p, GLP_MAX);
	samples_t *samples = env->samples;
	glp_add_cols(p, violation_idx(0, samples));
	glp_add_rows(p, violation_idx(1, samples));
	return p;
}


glp_prob *add_hyperplane(glp_prob *p, const env_t *env) {
	int hyperplane_cnt = env->samples->dimension + 1;
	// TODO default parms
	for (int i = 1; i <= hyperplane_cnt; i++) {
		glp_set_col_kind(p, i, GLP_CV);
		glp_set_col_bnds(p, i, GLP_FR, 0., 0.);
	}
	return p;
}

char label_to_varname(int label) {
	return label > 0 ? 'x' : 'y';
}

double label_to_obj(int label) {
	return label > 0 ? 1. : 0.;
}

double label_to_bound(int label, params_t *params) {
	return label > 0 ? 
		1. - params->epsilon_positive : 
		-params->epsilon_negative;
}

void add_sample(glp_prob *p, size_t class, size_t sample_index, 
		const env_t *env) {
	samples_t *samples = env->samples;
	int label = samples->label[class];
	char name[NAME_LEN_MAX];
	snprintf(name, NAME_LEN_MAX, "%c%u", 
			label_to_varname(label), 
			(unsigned int) sample_index + 1); 

	// Column
	int col_idx = idx(0, class, sample_index, samples);
	glp_set_col_name(p, col_idx, name);
	glp_set_col_kind(p, col_idx, GLP_BV);
	glp_set_obj_coef(p, col_idx, label_to_obj(label));

	// Row
	int row_idx = idx(1, class, sample_index, samples);
	glp_set_row_name(p, row_idx, name);
	params_t *params = env->params;
	glp_set_row_bnds(p, row_idx, GLP_UP, 0., label_to_bound(label, params));

	int dimension = samples->dimension;
	// Set coefficients of w
	sparse_vector_t *v = to_sparse(dimension, 
			samples->samples[class][sample_index], 2); 
	// Set coefficient of c
	append(v, dimension + 1, -1.); 
	// Change sign depending on sample class
	multiply(v, -label);
	// Add sample decision variable
	append(v, col_idx, label); 
	glp_set_mat_row(p, row_idx, v->len, v->ind, v->val);

	free(delete_sparse_vector(v));
}

glp_prob *add_samples(glp_prob *p, const env_t *env) {
	samples_t *samples = env->samples;
	for (size_t class = 0; class < samples->class_cnt; class++) {
		int cnt = samples->count[class];
		for (size_t idx = 0; idx < cnt; idx++) {
			add_sample(p, class, idx, env);
		}
	}

	return p;
}


glp_prob *add_precision(glp_prob *p, const env_t *env) {
	samples_t *samples = env->samples;
	int col_idx = violation_idx(0, samples);
	glp_set_col_name(p, col_idx, "V");
	glp_set_col_kind(p, col_idx, GLP_CV);
	params_t *params = env->params;
	glp_set_col_bnds(p, col_idx, params->violation_type, 0., 0.);
	glp_set_obj_coef(p, col_idx, -params->lambda);

	int row_idx = violation_idx(1, samples);
	glp_set_row_name(p, row_idx, "V");
	double theta = params->theta;
	glp_set_row_bnds(p, row_idx, GLP_UP, 0., 
			-theta * params->epsilon_precision);
	sparse_vector_t *constraint = precision_row(samples, theta);
	glp_set_mat_row(p, row_idx, 
			constraint->len, constraint->ind, constraint->val);
	free(delete_sparse_vector(constraint));

	return p;
}


glp_prob *milp(const env_t *env) {
	if (!is_binary(env->samples)) {
		return NULL;
	}
	glp_prob *p = init_prob(env);
	p = add_hyperplane(p, env);
	p = add_samples(p, env);
	p = add_precision(p, env);
	return p;
}
