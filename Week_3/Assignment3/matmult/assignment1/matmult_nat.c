#include <stdio.h>

void matmult_nat(int m,int n,int k,double *A,double *B,double *C){
	if(m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	
	for (int i=0;i<m;i++){
		for (int j=0;j<n;j++){
			C[i*n+j]=0.0;
			for (int l=0;l<k;l++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	}	
}
