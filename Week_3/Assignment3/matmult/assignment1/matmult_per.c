#include <stdio.h>

void matmult_mnk(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
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

void matmult_mkn(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	
	for (int pp=0;pp<(m*n);pp++){
		C[pp]=0.0;
	}
	
	
	for (int i=0;i<m;i++){
		for (int l=0;l<k;l++){
			for (int j=0;j<n;j++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	}
}

void matmult_nmk(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	for (int j=0;j<n;j++){
		for (int i=0;i<m;i++){
			C[i*n+j]=0.0;
			for (int l=0;l<k;l++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	} 
}

void matmult_nkm(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	
	for (int pp=0;pp<(m*n);pp++){
		C[pp]=0.0;
	}
	
	for (int j=0;j<n;j++){
		for (int l=0;l<k;l++){
			for (int i=0;i<m;i++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	}
}

void matmult_kmn(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	
	for (int pp=0;pp<(m*n);pp++){
		C[pp]=0.0;
	}
	
	for (int l=0;l<k;l++){
		for (int i=0;i<m;i++){
			for (int j=0;j<n;j++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	}
}

void matmult_knm(int m,int n,int k,double *A,double *B,double *C){
	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
		return;
	}
	
	for (int pp=0;pp<(m*n);pp++){
		C[pp]=0.0;
	}
	
	for (int l=0;l<k;l++){
		for (int j=0;j<n;j++){
			for (int i=0;i<m;i++){
				C[i*n+j]+=A[i*k+l]*B[l*n+j];
			}
		}
	}
}

