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
	
	// defining f and initializing first guess and initializing boundary conditions
	init(u,u_aux,f,N,start_T);

	double runtime = jacobi(u,u,f,N,iter_max, &tolerance);

	printf("%d,%d,%lf,%s\n",N, iter_max, runtime, "gpu_norm");
    
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

    return(0);
}
