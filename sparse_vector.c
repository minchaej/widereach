#include "widereach.h"
#include "helper.h"

void delete_sparse_vector(sparse_vector_t *v) {
	v->len = v->extra = 0;
	free(v->ind);
	free(v->val);
}

sparse_vector_t *to_sparse(size_t nmemb, double *ary, size_t extra) {
	sparse_vector_t *v = CALLOC(1, sparse_vector_t);
	size_t len = nmemb + extra + 1;
	v->ind = CALLOC(len, int);
	v->val = CALLOC(len, double);
	v->extra = extra;

	v->len = (int) nmemb;
	for (size_t i = 1; i <= nmemb; i++) {
		v->ind[i] = (int) i;
		v->val[i] = ary[i-1];
	}
	return v;
}

int append(sparse_vector_t *v, int ind, double x) {
	if (!v->extra) {
		return -1;
	}
	int idx = v->len + 1;
	v->ind[idx] = ind;
	v->val[idx] = x;
	v->len = idx;
	return (--v->extra);
}


sparse_vector_t *multiply(sparse_vector_t *v, double x) {
	for (int i = 1; i <= v->len; v++) {
		v->val[i] *= x;
	}
	return v;
}
