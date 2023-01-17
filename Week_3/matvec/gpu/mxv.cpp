#include "mxv.h"
#include <stdio.h>
#include <omp.h>

double mxv_single(int m, int n, double* a, double* b, double* c){
	if (m<=0 || n<=0 || a==NULL || b==NULL || c==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return NAN;
	}
	int i,j;
	double sum;
	double runtime;
	double runtimeWithLoad;

	double t = omp_get_wtime();
	#pragma omp target data map(to: b[0: m * n], c[0:m]) map(from: a[0:m]) 
	{
		double t = omp_get_wtime();
		#pragma omp target teams num_teams(108) //thread_limit(32)
		#pragma omp distribute private(i, j, sum) 
		for (i=0;i<m;i++){
			sum = 0.0;
			#pragma omp parallel 
				//printf("%d\n", omp_get_num_threads());
			#pragma omp for
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

double mxv_multi(int m, int n, double* a, double* b, double* c){
	if (m<=0 || n<=0 || a==NULL || b==NULL || c==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return NAN;
	}
	int i,j;
	double sum;
	double runtime_d0;
	double runtime_d1;
	double runtimeWithLoad;

	double t = omp_get_wtime();

	#pragma omp target data device(0) map(to: b[0: m * n], c[0:m]) map(from: a[0:m]) 
	{
		double t = omp_get_wtime();
		#pragma omp target teams distribute parallel for \
			num_teams(108 / 2) \
			private(i, j, sum) 
		for (i=0;i<m / 2;i++){
			sum = 0.0;
			for (j=0;j<n / 2;j++){
				sum += b[i*n+j]*c[j];
			}
			a[i] = sum;
		}
		runtime_d0 = omp_get_wtime() - t;
	}
	#pragma omp target data device(1) map(to: b[(m * n)/2: m * n], c[m / 2: m]) map(from: a[m / 2: m]) 
	{
		double t = omp_get_wtime();
		#pragma omp target teams distribute parallel for \
		num_teams(108 / 2) \
		private(i, j, sum) 
		for (i=m / 2;i<m;i++){
			sum = 0.0;
			for (j= n / 2;j<n;j++){
				sum += b[i*n+j]*c[j];
			}
			a[i] = sum;
		}
		runtime_d1 = omp_get_wtime() - t;
	}
	runtimeWithLoad = omp_get_wtime() - t;

	return runtimeWithLoad - runtime_d0 - runtime_d1;
}
