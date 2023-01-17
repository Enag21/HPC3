#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

double* matrix(size_t m, size_t n, double val);
double* vector(size_t m, double val);
void mxv(int m, int n, double* a, double* b, double* c);

int main(int argc, char *argv[]) {
	int i,j;
	//int S[]={1773, 1900, 2000, 2200, 2500, 3000, 3200, 3400, 3900, 4200, 4600, 5000, 5500, 6000};
	///*
	int S[60];
	int k = 100;
	for (i=0;i<=60;i++){
		S[i] = k;
		k += 100;
	}
	//*/
	int var1 = 1;
	int var2 = 2;
	int n = sizeof S / sizeof S[0];
	int I = 75;
	double memory, mflops, flPtOp;
	double start_t, end_t;
	double total_t;

	printf("Size\tMemory[kB]\tMflop/s\t\tCPUtime\n");
	
	for (i=0;i<n;i++){
		double* mat1 = matrix(S[i],S[i],var1);	
		double* vec1 = vector(S[i],var2);
		double* vec2 = vector(S[i],0);

		start_t = omp_get_wtime();
		for (j=0;j<I;j++){
			mxv(S[i],S[i], vec2, mat1, vec1);
		}
		end_t = omp_get_wtime();
		total_t = end_t - start_t;
		free(mat1);
		free(vec1);
		free(vec2);
	
		flPtOp = S[i] * (2.0 * S[i] - 1.0); // floating point operations: M(2N-1)
		mflops = 1e-6 * I * flPtOp / total_t;
		memory = ((S[i] * S[i] + 2*S[i]) * sizeof S[i]) / 1024.0; // M*M + 2*N * bytes / kB

		printf("%d\t%f\t%f\t%f\n",S[i],memory,mflops,total_t);
	}

    //double fpOperations{N * (2.0 * N - 1.0)}; 
    //std::cout << N << " " << ((N * N + N) * 8.0) / 1024.0 << " " << 1e-6 * ITERATIONS *(fpOperations / tcpu) <<  " " << tcpu << "\n";

    //mflops   = 1.0e-06 * nparts * loops;
    //memory = nparts * sizeof(particle_t);
    //memory /= 1024.0;	// in kbytes
	return(0);
}

double* matrix(size_t m, size_t n, double val){
	int i;
	if (m <= 0 || n <= 0)
	return NULL;

	double* A = malloc(m*n * sizeof(double));

	for(i=0; i< m*n; i++){
		A[i] = val;
	}
	return A;
}

double* vector(size_t m, double val){
	int i;
	if (m <= 0)
	return NULL;

	double* A = malloc(m * sizeof(double));

	for(i=0; i<m; i++){
		A[i] = val;
	}
	return A;
}

void mxv(int m, int n, double* a, double* b, double* c){
	if (m<=0 || n<=0 || a==NULL || b==NULL || c==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	int i,j;
	double sum;
	#pragma omp parallel for private(i,j,sum)
	for (i=0;i<m;i++){
		sum = 0.0;
		for (j=0;j<n;j++){
			sum += b[i*n+j]*c[j];
		}
		a[i] = sum;
		//printf("a[%d,%d] = %f\n",i,j,sum);
	}
}
