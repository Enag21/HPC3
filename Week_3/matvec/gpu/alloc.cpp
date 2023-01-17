#include "alloc.h"

double* matrix(size_t m, size_t n, double val){
	int i;
	if (m <= 0 || n <= 0)
	return NULL;

	double* A = (double*)malloc(m*n * sizeof(double));

	for(i=0; i< m*n; i++){
		A[i] = val;
	}
	return A;
}

double* vector(size_t m, double val){
	int i;
	if (m <= 0)
	return NULL;

	double* A = (double*)malloc(m * sizeof(double));

	for(i=0; i<m; i++){
		A[i] = val;
	}
	return A;
}
