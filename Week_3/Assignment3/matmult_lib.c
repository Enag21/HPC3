#include <cblas.h>
#include <stdio.h>

void matmult_lib(int m,int n,int k,double *A,double *B,double *C){

	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"Illegal input\n");
	}
	int lda=k,ldb=n,ldc=n;
	double alpha=1.0,beta=0.0;
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
