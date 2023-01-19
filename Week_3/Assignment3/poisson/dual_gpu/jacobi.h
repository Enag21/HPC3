#ifndef JACOBI_H
#define JACOBI_H

void jacobi_no_norm(double ***u0 ,double ***u_aux0 ,double ***f0,
				    double ***u1 ,double ***u_aux1 ,double ***f1, int N, int iter_max);

#endif