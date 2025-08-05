#ifndef FUNC_F
#define FUNC_F

#include <sys/time.h>

#define STR(s) #s
#define XSTR(s) STR(s)

#define TIMER_DEF(n)	 struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)	 gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)	 gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))
#define TIMER_PRINT(n) \
    do { \
        int rk;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);\
        if (rk==0) printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6);\
        fflush(stdout);\
        sleep(0.5);\
        MPI_Barrier(MPI_COMM_WORLD);\
    } while (0);

#define PRINT_RESULT_MATRIX(MAT, NAME, N,M) {    \
printf("%2s matrix:\n\t", NAME);        \
for (int i=0; i<N; i++) {               \
for (int j=0; j<M; j++)             \
printf("%4d ", MAT[i*N+j]);     \
printf("\n\t");                     \
}                                       \
printf("\n");                           \
}

#define PRINT_RESULT_VECTORF( V, NAME, LEN ) {    \
printf("%2s: ", NAME);                  \
for (int i=0; i<LEN; i++)               \
printf("%4f ", V[i]);               \
printf("\n");                           \
}

#define PRINT_RESULT_VECTORI( V, NAME, LEN ) {    \
printf("%2s: ", NAME);                  \
for (int i=0; i<LEN; i++)               \
printf("%4d ", V[i]);               \
printf("\n");                           \
}



double mu_fn_sol(double *v, int len);
double sigma_fn_sol(double *v, double mu, int len);

#endif
