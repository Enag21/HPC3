#include <stdio.h>
#include <omp.h>
#define MIN(x,y) ((x<y)?x:y)
#define BS 16

extern "C" 
{
    void matmult_blk_offload(int m,int n,int k,double *A,double *B,double *C, int bs)
    {
        if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	    }
        
        #pragma omp target enter data \
                map(to: A[0: m * k], B[0: k * n]) map(alloc: C[0:m * n])

        #pragma omp target teams distribute parallel for collapse(2)
        for (int i1 = 0; i1 < m; i1 += BS)
        {
            for (int j = 0; j < n; j++)
            {
                double sum[BS]={};
                for (int l = 0; l < k; l++)
                {
                    for (int i2 = 0; i2 < BS; i2++)
                    {
                        sum[i2] += A[(i1 + i2) * k + l] * B[l * n + j];
                    }
                }
                for(int i2 = 0; i2 < BS; i2++)
                {
                    C[(i1 + i2) * n + j] = sum[i2];
                }
            }
        }

        //
        int aux = m / BS;
        int it = m - aux * BS;
        if(it != 0)
        {
            for (int j = 0; j < n; j++)
            {
                double sum[BS]={};
                for (int l = 0; l < k; l++)
                {
                    for (int i2 = 0; i2 < it; i2++)
                    {
                        sum[i2] += A[(aux * BS + i2) * k + l] * B[l * n + j];
                    }
                }
                for(int i2 = 0; i2 < it; i2++)
                {
                    C[(aux * BS + i2) * n + j] = sum[i2];
                }
            }
        }
 

        #pragma omp target exit data \
                map(release: A[0: m * k], B[0: k * n]) map(from: C[0:m * n])
    }
}