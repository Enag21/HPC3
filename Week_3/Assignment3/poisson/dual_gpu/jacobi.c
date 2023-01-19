/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "jacobi.h"
#include <cublas.h>
//#include <cublas_v2.h>


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

void
jacobi_no_norm( double ***u0 ,double ***u_aux0 ,double ***f0, 
				double ***u1 ,double ***u_aux1 ,double ***f1, int N, int iter_max) {

	double h=2.0/(N+1.0);
	double pp=1.0/6.0;

	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)
	cudaSetDevice(0);
	
	for(int it = 0; it < iter_max; it++) 
	{
		// updating u for device (0)
		#pragma omp target teams distribute collapse(2) device(0) is_device_ptr(u0, u_aux0, u_aux1, f0) nowait
		for (int i=1;i<=N / 2;i++){
			for (int j=1;j<=N / 2;j++){

				double* x_1 = u_aux0[i - 1][j];
				double* x_2 = u_aux0[i + 1][j];
				if( i == N /2)
				{
					x_2 = u_aux1[0][j];
				}
				double* x_3 = u_aux0[i][j - 1];
				double* x_4 = u_aux0[i][j + 1];
				double* x_5 = u_aux0[i][j];
				double* x_6 = f0[i][j];
				double* x = u0[i][j];

				#pragma omp parallel for 
				for (int k=1;k<=N / 2;k++){

					x[k]=(x_1[k]+x_2[k]+x_3[k]+x_4[k]
					+x_5[k-1]+x_5[k+1]+h*h*x_6[k] )*pp;
				}
			}
		}
		// update u for device (1)
		#pragma omp target teams distribute collapse(2) device(1) is_device_ptr(u1, u_aux1, u_aux0, f1) nowait
		for (int i=0;i<=N / 2;i++){
			for (int j=1;j<=N / 2;j++){

				double* x_1 = u_aux1[i - 1][j];
				if (i == 0)
				{
					x_1 = u_aux0[N / 2][j];
				}
				double* x_2 = u_aux1[i + 1][j];
				double* x_3 = u_aux1[i][j - 1];
				double* x_4 = u_aux1[i][j + 1];
				double* x_5 = u_aux1[i][j];
				double* x_6 = f1[i][j];
				double* x = u1[i][j];

				#pragma omp parallel for 
				for (int k=1;k<=N / 2;k++){

					x[k]=(x_1[k]+x_2[k]+x_3[k]+x_4[k]
					+x_5[k-1]+x_5[k+1]+h*h*x_6[k] )*pp;
				}
			}
		}
		#pragma omp taskwait

		//device (0) 
		double*** tmp = u0;
		u0 = u_aux0;
		u_aux0 = tmp;

		//device (1)
		tmp = u1;
		u1 = u_aux1;
		u_aux1 = u1;
	}
	
	double*** tmp = u0;
	u0 = u_aux0;
	u_aux0 = tmp;

	//device (1)
	tmp = u1;
	u1 = u_aux1;
	u_aux1 = u1;
}
