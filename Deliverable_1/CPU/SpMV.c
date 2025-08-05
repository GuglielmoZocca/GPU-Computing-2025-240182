//CPU application code

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

//#define dtype int or float //Decide the type value of the matrix, input vector and output vector
#define NITER 10
//#define RAND //It is chosen the random format
//#define PRINT //Print the result
//#define SortC or SortR //Decide column or row sorting

#include "include/my_time_lib.h"

//Check correctness of macros

#if !defined(dtype)
#error "Must define value type (dtype=int, dtype=float, dtype=double)"
#endif

#if defined(SortC) && defined(SortR)
#error "Only a sort can be specified"
#endif

//Struct for the matrix sorting
typedef struct {
    int row;
    int col;
    double val;
} COOTuple;

//CPU SpMV implementation
void multiplicationCOO(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i;
    int prod;
    int row;

    for (i=0; i<nonZ; i++){

        row = COOR[i];
        R[row] += COOV[i]*V[COOC[i]];

    }
}

//CPU SpMV naive implementation
void multiplicationCOO_Naive(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){
    int i;
    int j;
    int prod;
    int row;

    for (i=0; i<nonZ; i++){

            row = COOR[i];
            R[row] += COOV[i]*V[COOC[i]];

    }
}

//Initialization function for row ans column coordinate vectors
void init_matrixI(int rows, int cols, int *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

//Initialization function for value coordinate vector, input and output vectors
void init_matrixV(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

//Support function for sorting by column
int compare_cooC(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->col != ib->col)
        return ia->col - ib->col;
    return ia->row - ib->row;
}

//Support function for sorting by row
int compare_cooR(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->row != ib->row)
        return ia->row - ib->row;
    return ia->col - ib->col;
}

//Matrix sort by row function
void sort_cooR(int *row, int *col, dtype *val, size_t nnz) {
    COOTuple *entries = (COOTuple *)malloc(nnz * sizeof(COOTuple));
    if (!entries) {
        perror("Failed to allocate memory for COO sorting");
        return;
    }

    for (size_t i = 0; i < nnz; i++) {
        entries[i].row = row[i];
        entries[i].col = col[i];
        entries[i].val = val[i];
    }

    qsort(entries, nnz, sizeof(COOTuple), compare_cooR);

    for (size_t i = 0; i < nnz; i++) {
        row[i] = entries[i].row;
        col[i] = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

//Matrix sort by column function
void sort_cooC(int *row, int *col, dtype *val, size_t nnz) {
    COOTuple *entries = (COOTuple *)malloc(nnz * sizeof(COOTuple));
    if (!entries) {
        perror("Failed to allocate memory for COO sorting");
        return;
    }

    for (size_t i = 0; i < nnz; i++) {
        entries[i].row = row[i];
        entries[i].col = col[i];
        entries[i].val = val[i];
    }

    qsort(entries, nnz, sizeof(COOTuple), compare_cooC);

    for (size_t i = 0; i < nnz; i++) {
        row[i] = entries[i].row;
        col[i] = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

//Support function for matrix randomization
int compare_rand(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

//Shuffle function
void shuffle(int *row,int *col,dtype *val,int nnz)
{
    if (nnz>1)
    {
        int i;
        int t1;
        int t2;
        int t3;
        int j;
        for (i = 0; i < nnz - 1; i++)
        {
            j = i + rand() / (RAND_MAX / (nnz - i) + 1);
            t1 = row[j];
            t2 = col[j];
            t3 = val[j];
            row[j] = row[i];
            col[j] = col[i];
            val[j] = val[i];
            row[i] = t1;
            col[i] = t2;
            val[i] = t3;
        }
    }
}

//Randomization function of the matrix
void initialize_random_coo(int *row, int *col, dtype *val, int nnz, int num_rows, int num_cols,int code) {
    printf("Randomization\n");

    int *usedR = (int *)malloc(num_rows*sizeof(int));
    char *usedC = (char *)malloc(num_cols*sizeof(char));
    int *usedC_i = (int *)malloc(num_cols*sizeof(int));
    if (!usedR) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    if (!usedC_i) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }

    int i;
    int j;

    for (i = 0; i < num_rows; i++) {
       usedR[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC[i] = 0;
    }
    for (i = 0; i < num_cols; i++) {
        usedC_i[i] = 0;
    }

    int r;
    int c;

    if(code == -1){
        srand(time(NULL));
    }else{
        srand(code);
    }
    size_t count = 0;
    while (count < nnz) {
        r = rand() % num_rows;
        if(usedR[r] < num_cols){
            usedR[r] = usedR[r] + 1;
            row[count] = r;
            count++;
        }
    }

    qsort(row, nnz, sizeof(int), compare_rand);

    int colum = 0;
    count = 0;
    while (count < nnz) {
        r = row[count];
        c = rand() % num_cols;

        if (!usedC[c]) {
            usedC[c] = 1;
            usedC_i[colum]=c;
            colum++;
            col[count] = c;
            val[count] = 1;

            if(count < nnz-1){
                if(r != row[count+1]){
                    for (i = 0; i < colum; i++) {
                        usedC[usedC_i[i]] = 0;
                        usedC_i[i] = 0;
                    }
                    colum = 0;
                }
            }
            count++;
        }

    }

    shuffle(row,col,val,nnz);

    free(usedR);
    free(usedC);
    free(usedC_i);
}

//Concatenation string function
char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}


int main(int argc, char *argv[]) {

  	#ifdef RAND

  	if (argc < 5) {
        printf("Usage: %s n m nonZero code\n", argv[0]);
        return(1);
    }

    printf("argv[0] = %s\n", argv[1]);
    printf("argv[1] = %s\n", argv[2]);
    printf("argv[2] = %s\n", argv[3]);
    printf("argv[3] = %s\n", argv[4]);

  	#else

    if (argc < 2) {
        printf("Usage: %s *.mtx\n", argv[0]);
        return(1);
    }

    printf("argv[0] = %s\n", argv[1]);

    FILE *fp;
    char *path = concat("../matrix/",argv[1]);

    fp = fopen(path, "r");
    if (fp == NULL) {
        printf("\nError; Cannot open file");
        exit(1);
    }

    free(path);

	#endif

    int i;
    int j;
    int n,m,nonZeros; //n: number of rows, m: number of columns, nonZeros: number of non-zeros
    dtype *b, *c; //b: input vector, c: output vector
    TIMER_DEF(0);
    double times[NITER];
    double mu = 0.0, sigma = 0.0; //mu: arithmetic mean of execution times, sigma: standard deviation of execution times

	#ifdef RAND

    int code = atoi(argv[4]); //Seed for randomization

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    nonZeros = atoi(argv[3]);

    #else

    char line[1024];

    //Read the dimension of the file
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    //Skip header and comments of *.mtx
    do {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "Error reading header\n");
            fclose(fp);
            return -1;
        }
    } while (line[0] == '%');

    // Read matrix size and number of non-zero elements
    if (sscanf(line, "%d %d %d", &n, &m, &nonZeros) != 3) {
        fprintf(stderr, "Invalid matrix size line\n");
        fclose(fp);
        return -1;
    }

    #endif

    printf("N: %d,M: %d,NonZero: %d\n", n,m,nonZeros);

    b=(dtype*)malloc(m*sizeof(dtype));
    c=(dtype*)malloc(n*sizeof(dtype));

    int *COOr = malloc(nonZeros*sizeof(int)); //Row vector of the COO format matrix
    int *COOc = malloc(nonZeros*sizeof(int)); //Column vector of the COO format matrix
    dtype *COOv = malloc(nonZeros*sizeof(dtype)); //Value vector of the COO format matrix


	int typ = (strcmp( XSTR(dtype) ,"int")==0); //Indicate the type of value: typ=true (int),type=false (others)

    printf("Fill matrix\n");

    #ifdef RAND

    init_matrixI(1, nonZeros, COOr, n);
    init_matrixI(1, nonZeros, COOc, m);
    init_matrixV(1, nonZeros, COOv, 0);

    initialize_random_coo(COOr, COOc, COOv, nonZeros, n, m,code);

    #else

    int row,col,val;

    char* buffer = malloc(size + 1);
    if (!buffer) {
        perror("Failed to allocate buffer");
        fclose(fp);
        return 1;
    }

    fread(buffer, 1, size, fp);
    char *save_outer;
    char *save_inner;
    buffer[size] = '\0';
    char *tokenOUT = strtok_r(buffer, "\n",&save_outer);
    char *tokenIN;

    for(i=0;i<nonZeros;i++) {
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
          tokenIN = strtok_r(NULL, " ",&save_inner);
		}
        tokenOUT = strtok_r(NULL, "\n",&save_outer);

    }

    free(buffer);

    #endif

    #ifdef SortR
    printf("SortR\n");
    sort_cooR(COOr,COOc,COOv,nonZeros);
    #endif
    #ifdef SortC
    printf("SortC\n");
    sort_cooC(COOr,COOc,COOv,nonZeros);
    #endif

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

    multiplicationCOO(COOr, COOc, COOv, b, c, nonZeros);

	#ifdef Print
   	if(typ){
   	    PRINT_RESULT_VECTORI(c, "C", n);
    }else{
        PRINT_RESULT_VECTORF(c, "C", n);
    }
    #endif

	init_matrixV(1,n,c,0);

    for (int k=-2; k<NITER; k++) {
	    TIMER_START(0);

    	multiplicationCOO(COOr, COOc, COOv, b, c, nonZeros);

        TIMER_STOP(0);
                
	    double iter_time = TIMER_ELAPSED(0) / 1.e6;
	    if( k >= 0) times[k] = iter_time;

        printf("Iteration %d tooks %lfs\n", k, iter_time);
        init_matrixV(1,n,c,0);
    }
    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");

    mu = mu_fn_sol(times, NITER);
    sigma = sigma_fn_sol(times, mu, NITER);

    int nflop = 2*nonZeros;

    unsigned int nMemAc = sizeof(dtype)*nonZeros*4 + sizeof(int)*nonZeros*2; //Number of bytes of memory access

    double flops = (nflop / mu)/1.e9; //GFLOPS

    double effBand = (nMemAc/1.e9)/mu; //Effective bandwidth (GB/s)

    double AI_O = 2;
    int AI_A = sizeof(dtype)*4 + sizeof(int)*2;
    double AI = AI_O/AI_A; //Arithmetic intensity

    //print test data
    printf("platf,matrix,id,n,m,nonZeros,Rand,sort,mu,sigma,nflop,nMemAc,AI_O,AI_A,AI,flops,effBand\n");
    printf("CPU,");
    #ifdef RAND
    printf("Random,");
    #else
    printf("%s,",argv[1]);
    #endif
    printf("COO,%d,%d,%d,",n,m,nonZeros);
    #ifdef RAND
    printf("yes,");
    #else
    printf("no,");
    #endif
    #ifdef SortC
    printf("SortC,");
    #else
    #ifdef SortR
    printf("SortR,");
    #else
    printf("no,");
    #endif
    #endif
    printf("%lf,%lf,%d,%u,%lf,%d,%lf,%lf,%lf",mu,sigma,nflop,nMemAc,AI_O,AI_A,AI,flops,effBand);

    free(COOr);
    free(COOc);
    free(COOv);
    free(b);
    free(c);

    #ifndef RAND

    fclose(fp);

    #endif

    return(0);
}

