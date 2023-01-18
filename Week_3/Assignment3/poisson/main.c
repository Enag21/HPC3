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

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
	int cores;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    
    #ifdef _OPENMP
    double t1,t2;
	#else
	clock_t t1,t2;
	#endif

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 7) {
	output_type = atoi(argv[5]);  // ouput type
	cores = atoi(argv[6]);
    }
    
	
    // allocate memory
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
	
	// Allocate memory on device
	double* u_data;
	double*** u_d = target_malloc_3d(N+2, N+2, N+2, &u_data);

	double* f_data;
	double*** f_d = target_malloc_3d(N+2, N+2, N+2, &f_data);

	double* u_aux_data;
	double*** u_aux_d = target_malloc_3d(N+2, N+2, N+2, &u_aux_data);


	#ifdef _OPENMP 
	t1=omp_get_wtime();
	#else
	t1=clock();
	#endif
	
	
	// defining f and initializing first guess and initializing boundary conditions
	init(u,u_aux,f,N,start_T);

	// Copying data from host to device
	omp_target_memcpy(	u_data, u[0][0], 
						(N + 2) * (N + 2) * (N + 2) * sizeof(double),
						0, 0, omp_get_default_device(), omp_get_initial_device());

	omp_target_memcpy(	u_aux_data, u_aux[0][0], 
					(N + 2) * (N + 2) * (N + 2) * sizeof(double),
					0, 0, omp_get_default_device(), omp_get_initial_device());

	omp_target_memcpy(	f_data, f[0][0], 
						(N + 2) * (N + 2) * (N + 2) * sizeof(double),
						0, 0, omp_get_default_device(), omp_get_initial_device());

	
	jacobi_no_norm(u_d,u_aux_d,f_d,N,iter_max);


	//printf("%d           %d                  %lf                  ",N,it,tolerance);

	omp_target_memcpy(u[0][0], u_data,
					(N + 2) * (N + 2) * (N + 2) * sizeof(double),
					0, 0, omp_get_default_device(), omp_get_initial_device());

	#ifdef _OPENMP
	t2=omp_get_wtime();
	printf("%lf             %d \n",t2-t1, cores);
	#else
	t2=clock();
	printf("%lf\n",(double) (t2-t1)/(CLOCKS_PER_SEC));
	#endif
    
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

	target_free_3d(u_d, u_data);
	target_free_3d(u_aux_d, u_aux_data);
	target_free_3d(f_d, f_data);

    return(0);
}
