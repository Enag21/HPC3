#include "mxv.h"
#include <stdio.h>
#include <omp.h>

double mxv(int m, int n, double* a, double* b, double* c){
	if (m<=0 || n<=0 || a==NULL || b==NULL || c==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	int i,j;
	double sum;
	double runtime;
	double runtimeWithLoad;

	double t = omp_get_wtime();
	#pragma omp target data map(to: b[0: m * n], c[0:m]) map(from: a[0:m])
	{
		double t = omp_get_wtime();
		#pragma omp target teams loop \
			num_teams(108) thread_limit(64) \
			map(to: b[0: m * n], c[0:m]) map(from: a[0:m]) \
			private(i, j, sum) 
		for (i=0;i<m;i++){
			sum = 0.0;
			for (j=0;j<n;j++){
				sum += b[i*n+j]*c[j];
			}
			a[i] = sum;
			//printf("a[%d,%d] = %f\n",i,j,sum);
		}
		runtime = omp_get_wtime() - t;
	}
	runtimeWithLoad = omp_get_wtime() - t;

	return runtimeWithLoad - runtime;
}
