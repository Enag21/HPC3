#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mandel_omp.h"
#include "writepng.h"



int
main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int*  image;
//    int*  image_d;
    int   dev_num = omp_get_default_device();

    width    = 4601;
    height   = 4601;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);


    image = (int *)malloc( width * height * sizeof(int));
    //image_d = (int*)omp_target_alloc(width * height * sizeof(int), dev_num);
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

    double dummy = 1.0;
    #pragma omp target data map(tofrom: dummy)
    {}

    mandel(width, height, image, max_iter);

    //omp_target_memcpy(image, image, width * height * sizeof(int), 0, 0, dev_num, omp_get_initial_device());

    //writepng("mandelbrot.png", image, width, height);

    //Remember to free!!

    //omp_target_free(image, dev_num);
    free(image);

    return(0);
}
