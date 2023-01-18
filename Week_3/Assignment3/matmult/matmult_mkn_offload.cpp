#include <stdio.h>
#include <omp.h>

extern "C" 
{
    void matmult_mkn_offload(int m,int n,int k,double *A,double *B,double *C)
    {
        if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	    }


        #pragma omp target enter data \
                map(to: A[0: m * k], B[0: k * n]) map(alloc: C[0:m * n])
    

        #pragma omp target teams distribute parallel for \
                num_teams(108) thread_limit(64) 
        for (int pp = 0;pp < (m * n); pp++)
        {
            C[pp] = 0.0;
        }
        
        #pragma omp target teams distribute parallel for num_teams(m) thread_limit(64)
        for (int i = 0; i < m; i++)
        {
            for (int l = 0; l < k; l++)
            {
                double a = A[i * k + l];
                for (int j = 0; j < n; j++)
                {
                    C[i * n + j] += a * B[l * n + j];
                }
            }
        }

        #pragma omp target exit data \
                map(release: A[0: m * k], B[0: k * n]) map(from: C[0:m * n])
    }
}