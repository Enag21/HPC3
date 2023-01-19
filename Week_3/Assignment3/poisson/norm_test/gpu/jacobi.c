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

	#pragma omp target teams distribute parallel for collapse(3) reduction(+: sum) is_device_ptr(a, b)
	for (int i=1;i<=N;i++){
		for (int j=1;j<=N;j++){
			for (int k=1;k<=N;k++){
				double x=a[i][j][k];
				double y=b[i][j][k];
				sum += (x-y)*(x-y);
			}
		}
	}
	return sqrt(sum);
}

double
jacobi(double ***u,double ***u_aux,double ***f,int N,int iter_max, double* tol) {

	double h=2.0/(N+1.0);
	double pp=1.0/6.0;
	double d = DBL_MAX;
	int it = 0;

	#pragma omp target  enter data \
		map(alloc: u[0:N + 2][0:N + 2][0:N + 2], u_aux[0:N + 2][0:N + 2][0:N + 2], f[0:N + 2][0:N + 2][0:N + 2])
	
	double t = omp_get_wtime();
	while ( d > *tol ||  it < iter_max)
	{	
		// updating u
		#pragma omp target teams distribute parallel for collapse(3)
		for (int i=1;i<=N;i++){
			for (int j=1;j<=N;j++){
				// #pragma omp parallel for 
				for (int k=1;k<=N;k++){

					u[i][j][k]=(u_aux[i - 1][j][k]+u_aux[i + 1][j][k]+u_aux[i][j - 1][k]+u_aux[i][j + 1][k]
					+u_aux[i][j][k-1]+u_aux[i][j][k+1]+h*h*f[i][j][k] )*pp;
				}
			}
		}
		double ***tmp = u;
		u = u_aux;
		u_aux = tmp;

		d = norm(u, u_aux, N);
		it++;
	}
	double ***tmp = u;
	u = u_aux;
	u_aux = tmp;
	double runtime = omp_get_wtime() - t;

	#pragma omp target exit data \
		map(release: u_aux[0:N + 2][0:N + 2][0:N + 2], f[0:N + 2][0:N + 2][0:N + 2]) map(from: u[0:N + 2][0:N + 2][0:N + 2])
	return runtime;
}