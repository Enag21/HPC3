extern "C"{
#include <cblas.h>
}
#include <stdio.h>
#include <cublas.h>
#include <cublas_v2.h>

extern "C"{
void matmult_lib(int m,int n,int k,double *A,double *B,double *C){

	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
	}
	int lda=k,ldb=n,ldc=n;
	double alpha=1.0,beta=0.0;
	cblas_dgemm(CblasRowMajor ,CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void matmult_lib_offload(int m,int n,int k,double *A,double *B,double *C){

	if (m<=0 || n<=0 || k<=0 || A==NULL || B==NULL || C==NULL){
		fprintf(stderr,"%s: Illegal input\n",__func__);
	}
	
	double *devPtrA;
	double *devPtrB;
	double *devPtrC;
	
	cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    
    // allocate memory in GPU
    cudaStat = cudaMalloc ((void**)&devPtrA, m*k*sizeof(*A));
    if (cudaStat != cudaSuccess) {
        fprintf (stderr,"%s: device memory allocation failed\n",__func__);
        return;
    }
    
	cudaStat = cudaMalloc ((void**)&devPtrB, k*n*sizeof(*B));
    if (cudaStat != cudaSuccess) {
        fprintf (stderr,"%s: device memory allocation failed\n",__func__);
        cudaFree(devPtrA);
        return;
    }
	
	cudaStat = cudaMalloc ((void**)&devPtrC, m*n*sizeof(*C));
    if (cudaStat != cudaSuccess) {
        fprintf(stderr,"%s: device memory allocation failed\n",__func__);
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        return;
    }
    
	// copy data from CPU to GPU
	stat = cublasSetMatrix (k,m,sizeof(*A),A,k,devPtrA,k);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"%s: data download failed\n",__func__);
        cudaFree(devPtrA);
        return;
    }
	
	stat = cublasSetMatrix (n,k,sizeof(*B),B,n,devPtrB,n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"%s: data download failed\n",__func__);
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        return;
    }
    
    
    // create CUDA handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"%s: CUBLAS initialization failed\n",__func__);
        return ;
    }
    
	double alpha=1.0,beta=0.0;
	
	
	// call cublas, A, B and C are colmajored in GPU, hence in GPU A^T, B^T and C^T is stored
	// therefore we want B^T * A^T = C^T
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, devPtrB, n, devPtrA, k, &beta, devPtrC, n);
	
	// copy date from GPU to CPU
    stat = cublasGetMatrix (m,n,sizeof(*C),devPtrC,m,C,m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"%s: data upload failed\n",__func__);
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
        cublasDestroy(handle);
        return;
    }
	
	
	// free memory from GPU and destroy handle
	cudaFree(devPtrA);
	cudaFree(devPtrB);
	cudaFree(devPtrC);
    cublasDestroy(handle);
	
}
}

