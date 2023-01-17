/* gaus_seidel.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif


double norm(double ***a,double ***b,int N){
	// function calculates norm between matrices
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

int
gauss_seidel(double ***u,double ***u_aux,double ***f,int N,int iter_max,double *tol) {

	int i,j,k;
	double d=DBL_MAX;
	int it=0;
	double h=2.0/(N+1.0);
	double pp=1.0/6.0;
	
	
	#ifndef _OPENMP
	while (d>*tol && it<iter_max){
	#else
	for (it=0;it<iter_max;it++){
	#endif
		
		#pragma omp parallel default(none) private(i,j,k) shared(h,pp,u,u_aux,f,N)
		{
			
		// copy u to u_aux
		#pragma omp for collapse(2)
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				double* aux_1 = u_aux[i][j];
				double* aux_2 = u[i][j];
				for (k=1;k<=N;k++){
					aux_1[k]=aux_2[k];
				}
			}
		}
		
		
		
		// updating u
		#pragma omp for ordered(3) // schedule(static,1)
		for (i=1;i<=N;i++){
			for (j=1;j<=N;j++){
				for (k=1;k<=N;k++){
					#pragma omp ordered depend(sink:i-1,j,k) depend(sink:i,j-1,k) depend(sink:i,j,k-1)
					//int t_id=omp_get_thread_num();
					//printf("thread = %d, (i,j,k) = (%d,%d,%d)\n",t_id,i,j,k);
					u[i][j][k]=(u[i-1][j][k]+u[i+1][j][k]+u[i][j-1][k]+u[i][j+1][k]+u[i][j][k-1]+u[i][j][k+1]+h*h*f[i][j][k])*pp;
					#pragma omp ordered depend(source)
				}
			}
		}
		
		} /* end of parallel */
		
		#ifndef _OPENMP
		it++;
		#endif
		d=norm(u,u_aux,N);
	}
	*tol=d;
	return(it);
}
