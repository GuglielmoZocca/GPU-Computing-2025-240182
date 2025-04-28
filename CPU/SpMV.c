#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#define dtype int
#define NITER 10
//#define RAND
#define COO
//#define PRINT


#include "include/my_time_lib.h"

typedef struct {
    int row;
    int col;
    double val;
} COOTuple;


void multiplicationCOO(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){
    int i;
    int j;
    int prod;
    //for (i=0; i<N; i++){
        for (j=0; j<nonZ; j++){

            //if(COOR[j] == i){

              R[COOR[j]] += COOV[j]*V[COOC[j]];

            //}

        }


   //}
}

/*void multiplicationCSR(dtype *CSRR, dtype *CSRC, dtype *CSRV, dtype *V, dtype *R, int N){
    int i;
    int n;
    int j;
    for (i=0; i<N; i++){

        n = CSRR[i+1] - CSRR[i];

        for (j=0; j<n; j++){

            R[i] += CSRV[j + CSRR[i]]*V[CSRC[j + CSRR[i]]];

        }

    }
}*/

void init_matrixI(int rows, int cols, int *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

void init_matrixV(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

/*void create_CSR(dtype *CSRRF,dtype *CSRR, dtype *CSRC, dtype *CSRV,int N,int nonZ) {

    int i;
    int j;


    init_matrix(1, N+1, CSRR, 0);

    for (i = 0; i < nonZ; i++){
      for (j = CSRRF[i]; j < N; j++) {
          CSRR[j+1]++;
      }
    }

    int tempr;
    int tempc;
    int tempv;
    for (i = 0; i < nonZ; i++) {
        int min = CSRRF[i];
        int indexM = i;
        for (int j = i; j < nonZ; j++) {

            if(CSRRF[j] < min){
                min = CSRRF[j];
                indexM = j;
            }

         }
        tempr = CSRRF[i];
        tempc = CSRC[i];
        tempv = CSRV[i];
        CSRRF[i] = min;
        CSRC[i] = CSRC[indexM];
        CSRV[i] = CSRV[indexM];
        CSRRF[indexM] = tempr;
        CSRC[indexM] = tempc;
        CSRV[indexM] = tempv;

    }


    int numE;
    int k;
    int num = 0;
    i = 0;
    while (num<N){

        numE = CSRR[num+1] - CSRR[num];

        for (j=i; j<(numE + i); j++){

            int indexM = j;
            int minc = CSRC[j];

            for (k = j; k<(numE + i); k++) {

                if(CSRC[k] < minc){
                    minc = CSRC[k];
                    indexM = k;
                }

            }
            tempc = CSRC[j];
            tempv = CSRV[j];
            CSRC[j] = CSRC[indexM];
            CSRV[j] = CSRV[indexM];
            CSRC[indexM] = tempc;
            CSRV[indexM] = tempv;
        }
        i += numE;
        num++;

    }

}

void Sort_COO(dtype *COOR,dtype *COOC, dtype *COOV, int nonZ){
	int i;
    int j;


    int tempr;
    int tempc;
    int tempv;
    for (i = 0; i < nonZ; i++) {
        int min = COOR[i];
        int indexM = i;
        for (int j = i; j < nonZ; j++) {

            if(COOR[j] < min){
                min = COOR[j];
                indexM = j;
            }

         }
        tempr = COOR[i];
        tempc = COOC[i];
        tempv = COOV[i];
        COOR[i] = min;
        COOC[i] = COOC[indexM];
        COOV[i] = COOV[indexM];
        COOR[indexM] = tempr;
        COOC[indexM] = tempc;
        COOV[indexM] = tempv;

    }

    for (i = 0; i < nonZ; i++) {
        int row = COOR[i];
        int min = COOC[i];
		int indexM = i;
        for (int j = i; j < nonZ; j++) {
            if(COOR[j] == row){
                if(COOC[j] < min){
                	min = COOC[j];
                	indexM = j;
           		}
            }else{
            	break;
            }

        }
        tempr = COOR[i];
        tempc = COOC[i];
        tempv = COOV[i];
        COOR[i] = COOR[indexM];
        COOC[i] = min;
        COOV[i] = COOV[indexM];
        COOR[indexM] = tempr;
        COOC[indexM] = tempc;
        COOV[indexM] = tempv;
    }


}*/

int compare_coo(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->row != ib->row)
        return ia->row - ib->row;
    return ia->col - ib->col;
}

void sort_coo(int *row, int *col, dtype *val, size_t nnz) {
    COOTuple *entries = malloc(nnz * sizeof(COOTuple));
    if (!entries) {
        perror("Failed to allocate memory for COO sorting");
        return;
    }

    for (size_t i = 0; i < nnz; i++) {
        entries[i].row = row[i];
        entries[i].col = col[i];
        entries[i].val = val[i];
    }

    qsort(entries, nnz, sizeof(COOTuple), compare_coo);

    for (size_t i = 0; i < nnz; i++) {
        row[i] = entries[i].row;
        col[i] = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

void initialize_random_coo(int *row, int *col, dtype *val, int nnz, int num_rows, int num_cols) {
    /*unsigned long int total = num_rows * num_cols;
    if (nnz > total) {
        fprintf(stderr, "Too many non-zero entries for given matrix size!\n");
        exit(EXIT_FAILURE);
    }*/

    //char *used = (char *)malloc(total);  // 0 = unused, 1 = used
    char (*used)[num_cols] = (char (*)[num_cols])malloc(sizeof(char[num_rows][num_cols]));
    if (!used) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }

    int i;
    int j;

    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            used[i][j] = 0;
        }
    }

    size_t count = 0;
    while (count < nnz) {
        int r = rand() % num_rows;
        int c = rand() % num_cols;
        //size_t index = IDX(r, c, num_cols);

        if (!used[r][c]) {
            used[r][c] = 1;
            row[count] = r;
            col[count] = c;
            val[count] = 1;
            count++;
        }
    }

    free(used);
}



int main(int argc, char *argv[]) {



  	#ifdef RAND

  	if (argc < 4) {
        printf("Usage: %s n m nonZero\n", argv[0]);
        return(1);
    }


    printf("argv[0] = %s\n", argv[1]);
    printf("argv[1] = %s\n", argv[2]);
    printf("argv[2] = %s\n", argv[3]);


  	#else

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

	#endif


    int i;
    int j;
    int n,m,nonZeros;
    dtype *b, *c;
    TIMER_DEF(0);
    double times[NITER];
    double mu = 0.0, sigma = 0.0;


	#ifdef RAND

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    nonZeros = atoi(argv[3]);

    #else


    fscanf(fp,"%d %d %d", &n,&m,&nonZeros);

    #endif

    printf("N: %d,M: %d,NonZero: %d\n", n,m,nonZeros);

    b=(dtype*)malloc(m*sizeof(dtype));
    c=(dtype*)malloc(n*sizeof(dtype));

    //COO format
    //#ifdef COO

    int *COOr = malloc(nonZeros*sizeof(int));
    int *COOc = malloc(nonZeros*sizeof(int));
    dtype *COOv = malloc(nonZeros*sizeof(dtype));


	int typ = (strcmp( XSTR(dtype) ,"int")==0);

    printf("Fill matrix\n");

    #ifdef RAND

    init_matrixI(1, nonZeros, COOr, n);
    init_matrixI(1, nonZeros, COOc, m);
    init_matrixV(1, nonZeros, COOv, 0);

    initialize_random_coo(COOr, COOc, COOv, nonZeros, n, m);

    /*int numero = 0;
    int r;
    int cl;

    time_t t;
    srand((unsigned) time(&t));

    int continua = 1;

    while (numero < nonZeros) {
      	continua = 1;
		r = rand()%n;
        cl = rand()%m;
        for (int l = 0; l < nonZeros; l++) {
          if(COOr[l] == r && COOc[l] == cl){
            continua = 0;
            break;
          }
        }

        if (continua) {

          COOr[numero] = r;
          COOc[numero] = cl;
          COOv[numero] = 1;

          numero++;

        }

    }*/

    #else

    //fill the matrix
    int row,col,val;


    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    char* buffer = malloc(size + 1);  // +1 for null terminator
    if (!buffer) {
        perror("Failed to allocate buffer");
        fclose(fp);
        return 1;
    }

    fread(buffer, 1, size, fp);
    char *save_outer;
    char *save_inner;
    buffer[size] = '\0';  // Null-terminate for convenience
    char *tokenOUT = strtok_r(buffer, "\n",&save_outer);
    char *tokenIN;
    tokenOUT = strtok_r(NULL, "\n",&save_outer);

    for(i=0;i<nonZeros;i++) {
      	//printf("%d \n",i);
        //fscanf(fp,"%d %d %d\n", &row,&col,&val);
        tokenIN = strtok_r(tokenOUT, " ",&save_inner);
		for(j=0;j<3;j++){
          if(j==0){
            COOr[i]=(atoi(tokenIN)-1);
          }
          if(j==1){
            COOc[i]=atoi(tokenIN)-1;
          }
          if(j==2){
            if (typ) {
            	COOv[i]=atoi(tokenIN);
            }else{
              	COOv[i]=atof(tokenIN);
            }
          }
			//printf("%d \n",atoi(tokenIN));
          tokenIN = strtok_r(NULL, " ",&save_inner);
		}
        tokenOUT = strtok_r(NULL, "\n",&save_outer);

    }
    //fclose(fp);

	//return 0;
    /*for(i=0;i<nonZeros;i++) {
      	//printf("%d \n",i);
        fscanf(fp,"%d %d %d\n", &row,&col,&val);
        COOr[i]=row-1;
        COOc[i]=col-1;
        COOv[i]=val;
    }*/

    #endif

    printf("SORT\n");

    sort_coo(COOr,COOc, COOv, nonZeros);


    //initialize the matrix

    init_matrixV(1,n,c,0);



    printf("initialize vector\n");
    //initialiaze the vector
    if (typ) {
        for (i=0; i<m; i++) {
            b[i] = 1;
        }
    } else {
        for (i=0; i<m; i++) {
            b[i] = 1.0;
        }
    }


    printf("calculate moltiplication\n");

    multiplicationCOO(COOr, COOc, COOv, b, c, nonZeros, n);

    //PRINT_RESULT_VECTOR(COOr, "COOr", nonZeros);

    //PRINT_RESULT_VECTOR(COOc, "COOc", nonZeros);

    //PRINT_RESULT_VECTOR(COOv, "COOv", nonZeros);

    //PRINT_RESULT_VECTOR(b, "B", m);
	#ifdef PRINT
   	PRINT_RESULT_VECTOR(c, "C", n);
    #endif

	init_matrixV(1,n,c,0);

    //calculate the moltiplication
    for (int k=-2; k<NITER; k++) {
	    TIMER_START(0);
        multiplicationCOO(COOr, COOc, COOv, b, c, nonZeros, n);
        TIMER_STOP(0);
                
	    double iter_time = TIMER_ELAPSED(0) / 1.e6;
	    if( k >= 0) times[k] = iter_time;

        printf("Iteration %d tooks %lfs\n", k, iter_time);
        init_matrixV(1,n,c,0);
    }
    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");
    //calculate the performance
    mu = mu_fn_sol(times, NITER);
    sigma = sigma_fn_sol(times, mu, NITER);

    printf(" %10s | %10s | %10s |\n", "v name", "mu(v)", "sigma(v)");
    printf(" %10s | %10f | %10f |\n", "time", mu, sigma);

    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    double nMemAc = 4*(5*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %lf bytes.\n", nMemAc);

    double flops = nflop / mu;
    printf("Matrix-vector moltiplication COO achieved %lf MFLOP/s\n", flops/1.e6);

    double effBand = (nMemAc/1.e9)/mu;
    printf("Matrix-vector moltiplication COO effective bandwidth is %lf GB/s\n", effBand);


    //#endif

    //COO format
    /*#ifdef CSR

    dtype *CSRr = malloc((n+1)*sizeof(dtype));
    dtype *CSRrF = malloc(nonZeros*sizeof(dtype));
    dtype *CSRc = malloc(nonZeros*sizeof(dtype));
    dtype *CSRv = malloc(nonZeros*sizeof(dtype));




    printf("Fill matrix\n");
    #ifdef RAND

    init_matrix(1, nonZeros, CSRrF, n);
    init_matrix(1, nonZeros, CSRc, m);
    init_matrix(1, nonZeros, CSRv, 0);


    int numero = 0;
    int r;
    int cl;

    time_t t;
    srand((unsigned) time(&t));

    int continua = 1;

    while (numero < nonZeros) {
      	continua = 1;
		r = rand()%n;
        cl = rand()%m;
        for (int l = 0; l < nonZeros; l++) {
          if(CSRrF[l] == r && CSRc[l] == cl){
            continua = 0;
            break;
          }
        }

        if (continua) {

          CSRrF[numero] = r;
          CSRc[numero] = cl;
          CSRv[numero] = 1;

          numero++;

        }

    }

    #else
    //fill the matrix
    int row,col,val;
    for(i=0;i<nonZeros;i++) {
        fscanf(fp,"%d %d %d\n", &row,&col,&val);
        CSRrF[i]=row-1;
        CSRc[i]=col-1;
        CSRv[i]=val;
    }

    #endif

    create_CSR(CSRrF,CSRr,CSRc,CSRv,n,nonZeros);

    //initialize the matrix
    init_matrix(1,n,c,0);

    printf("initialize vector\n");
    //initialiaze the vector
    int typ = (strcmp( XSTR(dtype) ,"int")==0);
    if (typ) {
        for (i=0; i<m; i++) {
            b[i] = 1;
        }
    } else {
        for (i=0; i<m; i++) {
            b[i] = 1.0;
        }
    }


    printf("calculate moltiplication\n");

    multiplicationCSR(CSRr,CSRc, CSRv, b, c, n);

    PRINT_RESULT_VECTOR(CSRrF, "CSRrF", nonZeros);

    PRINT_RESULT_VECTOR(CSRr, "CSRr", n+1);

    PRINT_RESULT_VECTOR(CSRc, "CSRc", nonZeros);

    PRINT_RESULT_VECTOR(CSRv, "CSRv", nonZeros);

    PRINT_RESULT_VECTOR(b, "B", m);

   	PRINT_RESULT_VECTOR(c, "C", n);

	init_matrix(1,m,c,0);

    //calculate the moltiplication
    for (int k=-2; k<NITER; k++) {
	    TIMER_START(0);
        multiplicationCSR(CSRr,CSRc, CSRv, b, c, n);
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

    int nflop = n+4*n*m;
    printf("\nMatrix-vector moltiplication CSR required n+4*n*m = %d floating point operations.\n", nflop);

    double nMemAc = 4*(2*n + 7*m*n);
    printf("\nMatrix-vector moltiplication CSR read and write %lf bytes.\n", nMemAc);

    double flops = nflop / mu;
    printf("Matrix-vector moltiplication CSR achieved %lf MFLOP/s\n", flops/1.e6);

    double effBand = (nMemAc/1.e9)/mu;
    printf("Matrix-vector moltiplication CSR effective bandwidth is %lf GB/s\n", effBand);



    #endif
	*/
    #ifndef RAND

    fclose(fp);

    #endif



    return(0);
}

