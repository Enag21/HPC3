#include <stdio.h>

 extern void matmult_mkn_omp(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}

    int i, j, l, pp;

    #pragma omp parallel shared(A, B, C, m, n, k) private(pp, i, l, j)
    {
        #pragma omp for
        for (pp = 0;pp < (m * n); pp++)
        {
            C[pp] = 0.0;
        }
        
        #pragma omp for collapse(2) nowait
        for (i = 0; i < m; i++)
        {
            for (l = 0; l < k; l++)
            {
                double a = A[i * k + l];
                for (j = 0; j < n; j++)
                {
                    C[i * n + j] += a * B[l * n + j];
                }
            }
        }
    }

}