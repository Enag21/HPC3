#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    // Check for available devices.
    if (omp_get_num_devices() < 1 || omp_get_default_device() < 0){
        fprintf(stderr, "No devices found. Run 'nvidia-smi' to check.\n");
        exit(1);
    }

    printf("Warming up device %i ... ", omp_get_default_device()); 
    fflush(stdout);

    double t = omp_get_wtime();

    // We force the creation of context on the device by allocating and
    // transferring a dummy variable. You will learn about the omp target data
    // construct and the map clause later.
    double dummy = 1.0;
    #pragma omp target data map(tofrom: dummy)
    {}

    printf("time = %3.2f seconds\n", omp_get_wtime() - t);
}
