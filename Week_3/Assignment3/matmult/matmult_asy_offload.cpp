#include <stdio.h>
#include <omp.h>
#define MIN(x,y) ((x<y)?x:y)
#define SPLITS 5
#define BS 1

extern "C" 
{
    void matmult_asy_offload(int m,int n,int k,double *A,double *B,double *C)
    {
        if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	    }
        
        #pragma omp target enter data map(to:B[0:k*n])//,A[0:m*k]) map(alloc:C[0:m*n])

		int n_rows=m/SPLITS;
		//printf("n_rows = %d\n",n_rows);

       	for (int s=0;s<SPLITS;s++){
       		// split A and C into chunks of size n_rows
       		
       		int startA=s*n_rows*k;
       		int startC=s*n_rows*n;
       		int istart=s*n_rows;
       		int iend=(s+1)*n_rows;
       		
       		//printf("startA = %d\n",startA);
       		//printf("startC = %d\n",startC);
      		//printf("iend = %d\n",iend);
      		//printf("jump = %d\n",n_rows*k);
      		
       		#pragma omp target teams distribute parallel for collapse(2) \
       			map(to:A[startA:n_rows*k]) map(from:C[startC:n_rows*n]) nowait
        	for (int i1 = istart; i1 < iend; i1 += BS)
        	{
            	for (int j = 0; j < n; j++) // 
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

       		int aux = iend / BS;
        	int it = iend - aux * BS;
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
       	
       	
       	}
       	
       	
       	//#pragma omp taskwait
		
        #pragma omp target exit data map(release:B[0:k*n])//,A[0:m*k]) map(from:C[0:m*n])
    }
}
