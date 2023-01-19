/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "jacobi.h"
#include <omp.h>


#ifdef _OPEN_MP
#include <omp.h>
#endif

double norm(double ***a,double ***b,int N){
	// function calculates norm between arrays
	double sum=0.0;
	int i,j,k;

	#pragma omp parallel for default(none) private(i,j,k) shared(a,b,N) reduction(+:sum) collapse(2)
	for (i=1;i<=N;i++){
		for (j=1;j<=N;j++){
			double* aux_1 = a[i][j];
			double* aux_2 = b[i][j];
			for (k=1;k<=N;k++){
				double x=aux_1[k];
				double y=aux_2[k];
				sum+=(x-y)*(x-y);
			}
		}
	}
	return sqrt(sum);
}

double
jacobi_no_norm(double ***u,double ***u_aux,double ***f,int N,int iter_max) {

	double h=2.0/(N+1.0);
	double pp=1.0/6.0;

	#pragma omp target  enter data \
		map(alloc: u[0:N + 2][0:N + 2][0:N + 2], u_aux[0:N + 2][0:N + 2][0:N + 2], f[0:N + 2][0:N + 2][0:N + 2])
	
	double t = omp_get_wtime();
	for(int it = 0; it < iter_max; it++) 
	{	
		// updating u
		#pragma omp target teams distribute collapse(2)
		for (int i=1;i<=N;i++){
			for (int j=1;j<=N;j++){

				double* x_1 = u_aux[i - 1][j];
				double* x_2 = u_aux[i + 1][j];
				double* x_3 = u_aux[i][j - 1];
				double* x_4 = u_aux[i][j + 1];
				double* x_5 = u_aux[i][j];
				double* x_6 = f[i][j];
				double* x = u[i][j];

				#pragma omp parallel for 
				for (int k=1;k<=N;k++){

					x[k]=(x_1[k]+x_2[k]+x_3[k]+x_4[k]
					+x_5[k-1]+x_5[k+1]+h*h*x_6[k] )*pp;
				}
			}
		}
		double ***tmp = u;
		u = u_aux;
		u_aux = tmp;
	}
	double ***tmp = u;
	u = u_aux;
	u_aux = tmp;
	double runtime = omp_get_wtime() - t;

	#pragma omp target exit data \
		map(release: u_aux[0:N + 2][0:N + 2][0:N + 2], f[0:N + 2][0:N + 2][0:N + 2]) map(from: u[0:N + 2][0:N + 2][0:N + 2])
	return runtime;
}