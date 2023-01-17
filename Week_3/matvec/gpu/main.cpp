#include <stdio.h>
#include <stdlib.h>
#include "mxv.h"
#include "alloc.h"
#ifdef _OPENMP
#include <omp.h>
#endif



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
	double memory, gflops, flPtOp;
	double start_t, end_t;
	double total_t;
	int flag;

	if (argc == 2){
		flag=atoi(argv[1]);
	}

	printf("%d\n", omp_get_num_devices());

	printf("Size\tMemory[kB]\tGflop/s\t\tCPUtime\t\tDataTransferTime\tRuntimeWithoutLoads\n");
	
	for (i=0;i<n;i++){
		double* mat1 = matrix(S[i],S[i],var1);	
		double* vec1 = vector(S[i],var2);
		double* vec2 = vector(S[i],0);
		double loadingTime = 0.0;

		if (flag==0){
			start_t = omp_get_wtime();
			for (j=0;j<I;j++){
				loadingTime += mxv_single(S[i],S[i], vec2, mat1, vec1);
			}
			end_t = omp_get_wtime();
		}
		else if (flag==1){
			start_t = omp_get_wtime();
			for (j=0;j<I;j++){
				loadingTime += mxv_multi(S[i],S[i], vec2, mat1, vec1);
			}
			end_t = omp_get_wtime();
		}
		else{
			fprintf(stderr,"Use this properly nimrod\n");
			return(1);
		}
		
		total_t = end_t - start_t;
		free(mat1);
		free(vec1);
		free(vec2);
	
		flPtOp = S[i] * (2.0 * S[i] - 1.0); // floating point operations: M(2N-1)
		gflops = 1e-9 * I * flPtOp / (total_t - loadingTime);
		memory = ((S[i] * S[i] + 2*S[i]) * sizeof S[i]) / 1024.0; // M*M + 2*N * bytes / kB

		printf("%d\t%f\t%f\t%f\t%f\t\t%f\n",S[i],memory,gflops,total_t, loadingTime, total_t - loadingTime);
	}

    //double fpOperations{N * (2.0 * N - 1.0)}; 
    //std::cout << N << " " << ((N * N + N) * 8.0) / 1024.0 << " " << 1e-6 * ITERATIONS *(fpOperations / tcpu) <<  " " << tcpu << "\n";

    //mflops   = 1.0e-06 * nparts * loops;
    //memory = nparts * sizeof(particle_t);
    //memory /= 1024.0;	// in kbytes
	return(0);
}


