#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#define dtype int
#define NITER 10

#include "include/my_time_lib.h"


void multiplication(dtype *A, dtype *B, dtype *C, int M, int N){
	int i;
  	int j;
  	for (i=0; i<N; i++){
		for (j=0; j<M; j++){
			C[i] += A[i*N + j]*B[j];
		}
    }

}

void init_matrix(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols +j] = val;
        }
    }
}



int main(int argc, char *argv[]) {


    if (argc < 2) {
        printf("Usage: %s *.mtx\n", argv[0]);
        return(1);
    }


    printf("argv[0] = %s\n", argv[1]);

    FILE *fp;
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("\nError; Cannot open file");
        exit(1);
    }

    int i;
    int j;
    int n,m,nonZeros;
    dtype *a, *b, *c;


    fscanf(fp,"%d %d %d", &n,&m,&nonZeros);

    printf("N: %d,M: %d,NonZero: %d\n", n,m,nonZeros);

    b=(dtype*)malloc(m*sizeof(dtype));
    a=(dtype*)malloc(n*m*sizeof(dtype));
    c=(dtype*)malloc(n*sizeof(dtype));

    //initialize the matrix
    init_matrix(n,m,a,0);

    init_matrix(1,n,c,0);

    printf("Fill matrix\n");
    //fill the matrix
    int row,col,val;
    for(i=0;i<nonZeros;i++) {
        fscanf(fp,"%d %d %d\n", &row,&col,&val);
        a[(row-1)*n+(col-1)]=val;
    }

    printf("initialize vector\n");
    //initialiaze the vector
    time_t t;
    srand((unsigned) time(&t));
    int typ = (strcmp( XSTR(dtype) ,"int")==0);
    if (typ) {
        int rand_range = (1<<11);
        for (i=0; i<m; i++) {
            b[i] = rand()/(rand_range);
        }
    } else {
        for (i=0; i<m; i++) {
            b[i] = (dtype)rand()/((dtype)RAND_MAX);
        }
    }

    TIMER_DEF(0);
    double times[NITER];
    double mu = 0.0, sigma = 0.0;

    printf("calculate moltiplication\n");

    multiplication(a,b,c,m,n);

    PRINT_RESULT_MATRIX(a, "A", n,m);

    PRINT_RESULT_VECTOR(b, "B", m);

   	PRINT_RESULT_VECTOR(c, "C", n);

	init_matrix(1,m,c,0);

    //calculate the moltiplication
    for (int k=-2; k<NITER; k++) {
	    TIMER_START(0);
		multiplication(a,b,c,m,n);
        TIMER_STOP(0);
                
	    double iter_time = TIMER_ELAPSED(0) / 1.e6;
	    if( k >= 0) times[k] = iter_time;

        printf("Iteration %d tooks %lfs\n", k, iter_time);
        init_matrix(1,n,c,0);
    }
    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");
    //calculate the performance
    mu = mu_fn_sol(times, NITER);
    sigma = sigma_fn_sol(times, mu, NITER);

    printf(" %10s | %10s | %10s |\n", "v name", "mu(v)", "sigma(v)");
    printf(" %10s | %10f | %10f |\n", "time", mu, sigma);

    int nflop = 2*n*m;
    printf("\nMatrix-vector moltiplication required 2*n*m = %d floating point operations.\n", nflop);

    double nMemAc = ((2+1)*4*n*m);
    printf("\nMatrix-vector moltiplication read and write %lf bytes.\n", nMemAc);

    double flops = nflop / mu;
    printf("Matrix-vector moltiplication achieved %lf MFLOP/s\n", flops/1.e6);

    double effBand = (nMemAc/1.e9)/mu;
    printf("Matrix-vector moltiplication effective bandwidth is %lf GB/s\n", effBand);

    fclose(fp);
    return(0);
}
