#include <stdio.h>

void init(double ***u,double ***u_aux,double ***f,int N,double start_T){

	int nlim=N+1;
	double h=2.0/(N+1.0); // distance between points in grid
	int i,j,k;
	
	#pragma omp parallel default(none) private(i,j,k) shared(u,u_aux,f,start_T,N,nlim,h)
	{
	
	// initial guess of interior points
	#pragma omp for 
	for (i=1;i<=N;i++){	
		double** x = u[i];
		for (j=1;j<=N;j++){
			for (k=1;k<=N;k++){
				x[j][k]=start_T;
			}
		}
	}	

	// defining f_{ijk}
	
	#pragma omp for
	 for (i=0;i<=nlim;i++){
	 	double x = -1.0 + (double) i*h;
		int xm = x<=-3.0/8.0;
		double** fx = f[i];
		for (j=0;j<=nlim;j++){
			double y = -1.0 + (double) j*h;
			int ym = (y<=-0.5);
			for (k=0;k<=nlim;k++){
				double z = -1.0 + (double) k*h;
				fx[j][k]=-200.0*(xm && ym && -2.0/3.0<=z && z<=0);
			}
		}
	 }
	 
	 // boudary conditions
	 #pragma omp for 
	 for (i=0;i<=nlim;i++){
	 	for (j=0;j<=nlim;j++){
	 		u[i][nlim][j]=20.0;
			u[nlim][i][j]=u[0][i][j]=20.0;
			u[i][j][0]=u[i][j][nlim]=20.0;
			u[i][0][j]=0.0;
			u_aux[i][nlim][j]=20.0;
			u_aux[nlim][i][j]=u_aux[0][i][j]=20.0;
			u_aux[i][j][0]=u_aux[i][j][nlim]=20.0;
			u_aux[i][0][j]=0.0;
	 	}
	}
	
	} /* end of parallel region */
}
