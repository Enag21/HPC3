void
mandel(int disp_width, int disp_height, int *array, int max_iter) {

    double 	scale_real, scale_imag;
    double 	x, y, u, v, u2, v2;
    int 	i, j, iter;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;

	/*
	#pragma omp parallel for default(none) \
		shared(disp_width, disp_height, scale_real, scale_imag, max_iter, array) \
		private(i, j, x, y, u, v, u2, v2, iter) \
		schedule(dynamic, 5) \
		collapse(2)
	*/
	/*
	#pragma omp target teams distribute parallel for default(none) \
		shared(disp_width, disp_height, scale_real, scale_imag, max_iter, array) \
		private(i, j, x, y, u, v, u2, v2, iter) \
		num_teams(10) thread_limit(64) \
		is_device_ptr(array) \
		collapse(2)
	*/

	#pragma omp target teams distribute parallel \
		num_teams(108) thread_limit(64) \
		map(from: array[0:disp_height * disp_width]) \
		collapse(2)
    for(i = 0; i < disp_width; i++) {
		for(j = 0; j < disp_height; j++) {

			x = ((double)i * scale_real) - 2.25; 
			y = ((double)j * scale_imag) - 1.75; 

			u    = 0.0;
			v    = 0.0;
			u2   = 0.0;
			v2   = 0.0;
			iter = 0;

			while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
			v = 2 * v * u + y;
			u = u2 - v2 + x;
			u2 = u*u;
			v2 = v*v;
			iter = iter + 1;
			}

			// if we exceed max_iter, reset to zero
			iter = iter == max_iter ? 0 : iter;

			array[i*disp_height + j] = iter;
		}
    }
}
