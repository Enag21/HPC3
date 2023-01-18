#include <stdio.h>
#define MIN(x,y) ((x<y)?x:y)

extern "C"
{
	void matmult_blk_omp(int m,int n,int k,double *A,double *B,double *C, int bs){
		if(m<=0 || n<=0 || k<=0 || bs<=0 || A==NULL || B==NULL || C==NULL){
			fprintf(stderr,"%s: Illegal input\n",__func__);
			return;
		}
		
		#pragma omp parallel// default(none) shared(m,n,k,A,B,C,bs)
		{
		
		#pragma omp for
		for (int pp=0;pp<(m*n);pp++){
			C[pp]=0.0;
		}
		
		#pragma omp for
		for (int i1=0;i1<m;i1+=bs){
			int ilim=MIN(m-i1,bs);
			for (int l1=0;l1<k;l1+=bs){
				int llim=MIN(k-l1,bs);
				for (int j1=0;j1<n;j1+=bs){
					int jlim=MIN(n-j1,bs);
					for (int j2=0;j2<jlim;j2++){
						for (int l2=0;l2<llim;l2++){
							for (int i2=0;i2<ilim;i2++){
								C[(i1+i2)*n+j1+j2]+=A[(i1+i2)*k+l1+l2]*B[(l1+l2)*n+j1+j2];
							}
						}
					}
				}
			}
		}
		
		}  // end of parallel
	}
}


