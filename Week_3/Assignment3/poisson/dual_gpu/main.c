/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif

#include "init.h"
#include "jacobi.h"
#include <cublas.h>
#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
	int cores=0;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double t1,t2;

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 7) {
	output_type = atoi(argv[5]);  // ouput type
	cores = atoi(argv[6]);
    }

// -----------------------------------------
//
//    Allocate memory on host
//
// -----------------------------------------
    
    if ( (u = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    double ***f=malloc_3d(N+2,N+2,N+2);
    if (f==NULL){
    	free_3d(u);
    	perror("array u: allocation failed");
        exit(-1);
	}
	
	double ***u_aux=malloc_3d(N+2,N+2,N+2);
	if (u_aux==NULL){
		free_3d(f);
		free_3d(u);
		perror("array u: allocation failed");
    	exit(-1);
	}
	//printf("%d\n", omp_get_num_devices());
	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)
	cudaSetDevice(0);
	printf("check1\n");
// -----------------------------------------
//
//    Allocate memory on device (0)
//
// -----------------------------------------
	omp_set_default_device(0);
	double* u_data0;
	double*** u_d0 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &u_data0);

	double* f_data0;
	double*** f_d0 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &f_data0);

	double* u_aux_data0;
	double*** u_aux_d0 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &u_aux_data0);

	printf("check\n");
// -----------------------------------------
//
//    Allocate memory on device (1)
//
// -----------------------------------------
	omp_set_default_device(1);
	double* u_data1;
	double*** u_d1 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &u_data1);

	double* f_data1;
	double*** f_d1 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &f_data1);

	double* u_aux_data1;
	double*** u_aux_d1 = target_malloc_3d((N+2) / 2, (N+2), (N+2), &u_aux_data1);

// -----------------------------------------
//
//    Defining f and initializing first guess and initializing boundary conditions on host
//
// -----------------------------------------
	printf("check2\n");
	
	init(u,u_aux,f,N,start_T);


// -----------------------------------------
//
//    Copying data from host to device (0)
//
// -----------------------------------------
	omp_set_default_device(0);
	omp_target_memcpy(	u_data0, u[0][0], 
						(((N + 2) * (N + 2) * (N + 2)) / 2)* sizeof(double),
						0, 0, omp_get_default_device(), omp_get_initial_device());


	omp_target_memcpy(	u_aux_data0, u_aux[0][0], 
					(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
					0, 0, omp_get_default_device(), omp_get_initial_device());

	omp_target_memcpy(	f_data0, f[0][0], 
						(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
						0, 0, omp_get_default_device(), omp_get_initial_device());
	
	printf("check3\n");
// -----------------------------------------
//
//    Copying data from host to device (1)
//
// -----------------------------------------
// We need to offset the memory copied in device (0)
	omp_set_default_device(1);
	omp_target_memcpy(	u_data1, u[0][0], 
						(((N + 2) * (N + 2) * (N + 2)) / 2)* sizeof(double),
						0, (((N + 2) * (N + 2) * (N + 2)) / 2), omp_get_default_device(), omp_get_initial_device());


	omp_target_memcpy(	u_aux_data1, u_aux[0][0], 
					(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
					0, (((N + 2) * (N + 2) * (N + 2)) / 2), omp_get_default_device(), omp_get_initial_device());

	omp_target_memcpy(	f_data1, f[0][0], 
						(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
						0, (((N + 2) * (N + 2) * (N + 2)) / 2), omp_get_default_device(), omp_get_initial_device());

	printf("check4\n");
	
// -----------------------------------------
//
//    Run jacobi algorithm on device with dual GPUs
//
// -----------------------------------------
	t1=omp_get_wtime();
	jacobi_no_norm( u_d0 ,u_aux_d0 ,f_d0, 
					u_d1 ,u_aux_d1 ,f_d1, N,iter_max);
	t2=omp_get_wtime();


// -----------------------------------------
//
//    Copy memory back to host
//
// -----------------------------------------
	omp_set_default_device(0);
	omp_target_memcpy(u[0][0], u_data0,
					(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
					0, 0, omp_get_initial_device(), omp_get_default_device());

	// offset destination memory
	omp_set_default_device(1);
	omp_target_memcpy(u[0][0], u_data0,
					(((N + 2) * (N + 2) * (N + 2)) / 2) * sizeof(double),
					(((N + 2) * (N + 2) * (N + 2)) / 2), 0, omp_get_initial_device(), omp_get_default_device());



	printf("%d,%d,%lf,%s\n",N, iter_max, t2-t1, "dual_gpu");
    
    
    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s\n", output_filename);
	    print_binary(output_filename, N+2, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s\n", output_filename);
	    print_vtk(output_filename, N+2, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free_3d(f);
    free_3d(u_aux);
    free_3d(u);

	omp_set_default_device(0);
	target_free_3d(u_d0, u_data0);
	target_free_3d(u_aux_d0, u_aux_data0);
	target_free_3d(f_d0, f_data0);

	omp_set_default_device(1);
	target_free_3d(u_d1, u_data1);
	target_free_3d(u_aux_d1, u_aux_data1);
	target_free_3d(f_d1, f_data1);

    return(0);
}
