/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "jacobi.h"
#include <cublas.h>
//#include <cublas_v2.h>
#include <omp.h>

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
		//omp_set_default_device(0);
		#pragma omp target is_device_ptr(u0, u_aux0, f0) device(0) nowait
		{
			#pragma omp teams distribute parallel for collapse(3)  
			for (int i=1; i < ((N + 2) / 2) - 1;i++){
				for (int j=1;j<=N;j++){
					for (int k=1;k<=N;k++){
						u0[i][j][k]=(u_aux0[i - 1][j][k]+u_aux0[i + 1][j][k]+u_aux0[i][j - 1][k]+u_aux0[i][j + 1][k]
						+u_aux0[i][j][k-1]+u_aux0[i][j][k+1]+h*h*f0[i][j][k] )*pp;
					}
				}
			}
			//printf("%d\n", omp_get_default_device());
		}

		int i_d0 = ((N + 2) / 2) - 1; // index for last layer in gpu 0
		int i_d1 = 0; // index in last layer for gpu 1

		#pragma omp target teams distribute is_device_ptr(u0, u_aux0, f0, u1, u_aux1, f1) nowait
		for(int j = 1; j <= N; j++)
		{
			#pragma omp parallel for
			for(int k = 1; k <= N; k++)
			{
				u0[i_d0][j][k] = pp *(u_aux0[i_d0 - 1][j][k] + u_aux1[0][j][k] + u_aux0[i_d0][j - 1][k] + u_aux0[i_d0][j + 1][k] 
											+ u_aux0[i_d0][j][k - 1] + u_aux0[i_d0][j][k + 1] + h * h *f0[i_d0][j][k]);
				u1[i_d1][j][k] = pp * (u_aux0[i_d0][j][k] + + u_aux1[i_d1 + 1][j][k] + u_aux1[i_d1][j - 1][k] + u_aux1[i_d1][j + 1][k] 
											+ u_aux1[i_d1][j][k - 1] + u_aux1[i_d1][j][k + 1] + h * h *f1[i_d1][j][k]);
			}
		}

		//omp_set_default_device(1);
		// update u for device (1)
		#pragma omp target is_device_ptr(u1, u_aux1, f1) device(1) nowait
		{
			#pragma omp teams distribute parallel for collapse(3) 
			for (int i=1;i < ((N + 2) / 2) - 1;i++){
				for (int j=1;j <= N;j++){
					for (int k=1;k<=N;k++){
						u1[i][j][k]=(u_aux1[i - 1][j][k]+u_aux1[i + 1][j][k]+u_aux1[i][j - 1][k]+u_aux1[i][j + 1][k]
						+u_aux1[i][j][k-1]+u_aux1[i][j][k+1]+h*h*f1[i][j][k] )*pp;
					}
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
