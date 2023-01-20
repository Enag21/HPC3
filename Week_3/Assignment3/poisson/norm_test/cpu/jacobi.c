/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <float.h>
#include <stdio.h>


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

int
jacobi(double ***u,double ***u_aux,double ***f,int N,int iter_max,double *tol) {

	int i,j,k;
	double h=2.0/(N+1.0);
	double pp=1.0/6.0;
	double d=DBL_MAX;
	int it=0;
	
	while (it<iter_max){
		
		// updating u
		#pragma omp parallel for shared(u,u_aux,N,h,f,pp) collapse(2)
		for (int i=1;i<=N;i++){
			for (int j=1;j<=N;j++){
				for (int k=1;k<=N;k++){

					u[i][j][k]=(u_aux[i - 1][j][k]+u_aux[i + 1][j][k]+u_aux[i][j - 1][k]+u_aux[i][j + 1][k]
					+u_aux[i][j][k-1]+u_aux[i][j][k+1]+h*h*f[i][j][k] )*pp;
				}
			}
		}

		double ***tmp = u;
		u = u_aux;
		u_aux = tmp;
		d=norm(u,u_aux,N);
		it++;
	}
	double ***tmp = u;
	u = u_aux;
	u_aux = tmp;
	*tol=d;
	return it;
}
