#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

//#define dtype int or float //Decide the type value of the matrix, input vector and output vector
#define NITER 10
//#define RAND //It is chosen the random format
//#define COO1 or COO2 or COO3 or COO4 or COO5 //Type of solution
//#define Pinned or Managed //Type of memory copy
//#define SortC or SortR //Decide column or row sorting
//#define PRINT //Print the result
//#define Check //Abilitate the check for the correctness of the solution through the CPU SpMV (only for integer matrix)

#define PFP32 10300 //Peak FP32 Compute Performance
//define PFP32 91600 //Peak FP32 Compute Performance
#define MB 933 //Peak Memory Bandwidth
//#define MB 864 //Peak Memory Bandwidth

//Check correctness of macros

#if !defined(dtype)
#error "Must define value type (dtype=int, dtype=float, dtype=double)"
#endif

#if !defined(COO1) && !defined(COO2) && !defined(COO3) && !defined(COO4) && !defined(COO5)
#error "The algorithm is not defined (COO1, COO2, COO3, COO4, COO5)"
#endif


#if !defined(Pinned) && !defined(Managed)
#error "Memcopy no specified (Pinned, Managed)"
#endif

#if (defined(COO1) && (defined(COO2) || defined(COO3) || defined(COO4) || defined(COO5))) || (defined(COO2) && (defined(COO1) || defined(COO3) || defined(COO4) || defined(COO5))) || (defined(COO3) && (defined(COO1) || defined(COO2) || defined(COO4) || defined(COO5))) || (defined(COO4) && (defined(COO1) || defined(COO2) || defined(COO3) || defined(COO5))) || (defined(COO5) && (defined(COO1) || defined(COO2) || defined(COO3) || defined(COO4)))
#error "Only an algorithm can be defined"
#endif

#if defined(Pinned) && defined(Managed)
#error "Only a memcopy can be specified"
#endif

#if defined(SortC) && defined(SortR)
#error "Only a sort can be specified"
#endif

#if defined(COO3) && (defined(SortC) || !defined(SortR))
#error "Not appropriate sorted for the algorithm (only SortR)"
#endif

#if defined(COO2) && (defined(SortC) || !defined(SortR))
#error "Not appropriate sorted for the algorithm (only SortR)"
#endif

#if (defined(COO2) && defined(Managed)) || (defined(COO3) && defined(Managed)) || (defined(COO4) && defined(Managed)) || (defined(COO5) && defined(Managed))
#error "The algorithm is not compatible with the Managed option (only Pinned)"
#endif

#if defined(COO1) && defined(Pinned)
#error "The algorithm is not compatible with the Pinned option (only Managed)"
#endif

#include "include/my_time_lib.h"

//Struct for the matrix sorting
typedef struct {
    int row;
    int col;
    double val;
} COOTuple;

//CPU SpMV implementation
void multiplicationCOO(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i;
    int row;
    for (i=0; i<nonZ; i++){

            row = COOR[i];
            R[row] += COOV[i]*V[COOC[i]];

    }
}

//GPU Element-wise Multiplication (GPU Kernel)
__global__
void multiplicationCOO1(int *COOC, dtype *COOV, dtype *V, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i < nonZ){

        COOV[i] = COOV[i]*V[COOC[i]];

    }

}

//GPU Matrix-Vector Multiplication with Row Reduction (GPU Kernel)
__global__
void multiplicationCOO2(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i < nonZ){

  		if(i == 0 || (COOR[i - 1] != COOR[i])){
    		dtype prod = 0;
            char continu = 0;
    		int row = COOR[i];
    		for(int j=i; j<nonZ; j++){

        		if(COOR[j] == row){

        		    prod += COOV[j]*V[COOC[j]];
            		continu = 1;

        		}else{

          			if(continu){
            			break;

          			}

        		}
    		}

    		R[row] = prod;
  		}
   }
}

//GPU SpMV via Row-wise Accumulation (One Thread per Row)
__global__
void multiplicationCOO3(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i < N){
		dtype prod = 0;
    	char continu = 0;

        for(int j=0; j<nonZ; j++){

            if(COOR[j] == i){

                prod += COOV[j]*V[COOC[j]];
                continu = 1;

            }else{

                  if(continu){

                    break;

                  }

            }
        }

        R[i] = prod;
    }


}

//GPU SpMV with Atomic Accumulation (GPU Kernel)
__global__
void multiplicationCOO4(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < nonZ) {

        atomicAdd(&R[COOR[i]], COOV[i]*V[COOC[i]]);

    }
}

//GPU SpMV Shared Memory Optimized COO (GPU Kernel)
__global__
void multiplicationCOO5(int *COOR, int *COOC, dtype *COOV,dtype *V, dtype *R, int nnz) {
  	extern __shared__ char shared_mem[];
    int *s_rows = (int*)shared_mem;
    dtype *s_vals = (dtype*)&s_rows[blockDim.x];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    if (tid >= nnz) return;

    s_rows[local_tid] = COOR[tid];
    s_vals[local_tid] = COOV[tid]*V[COOC[tid]];
    __syncthreads();

    if (local_tid > 0 && s_rows[local_tid] == s_rows[local_tid - 1]) return;

    dtype sum = s_vals[local_tid];
    int row = s_rows[local_tid];

    for (int i = local_tid + 1; i < blockDim.x && tid + (i - local_tid) < nnz; i++) {
        if (s_rows[i] == row) {
            sum += s_vals[i];
        } else {
            break;
        }
    }

    atomicAdd(&R[row], sum);
}

//Initialization function for row ans column coordinate vectors
void init_matrixV(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

//Initialization function for value coordinate vector, input and output vectors
void init_matrixI(int rows, int cols, int *matrix, int val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

//Support function for sorting by row
int compare_cooR(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->row != ib->row)
        return ia->row - ib->row;
    return ia->col - ib->col;
}

//Support function for sorting by column
int compare_cooC(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->col != ib->col)
        return ia->col - ib->col;
    return ia->row - ib->row;
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
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}



int main(int argc, char *argv[]) {

  	#ifdef RAND

  	if (argc < 6) {
        printf("Usage: %s n m nonZero blockSize code\n", argv[0]);
        return(1);
    }


    printf("argv[0] = %s\n", argv[1]);
    printf("argv[1] = %s\n", argv[2]);
    printf("argv[2] = %s\n", argv[3]);
    printf("argv[3] = %s\n", argv[4]);
    printf("argv[4] = %s\n", argv[5]);


  	#else

    if (argc < 3) {
        printf("Usage: %s *.mtx blockSize\n", argv[0]);
        return(1);
    }


    printf("argv[0] = %s\n", argv[1]);
    printf("argv[1] = %s\n", argv[2]);

    FILE *fp;

    char *path = concat("../matrix/",argv[1]);

    fp = fopen(path, "r");
    if (fp == NULL) {
        printf("\nError; Cannot open file");
        exit(1);
    }

    free(path);

	#endif

	#ifdef COO1
	TIMER_DEF(0);
	#endif

    int i;
    int j;
    int n,m,nonZeros; //n: number of rows, m: number of columns, nonZeros: number of non-zeros
    dtype *b, *c; //b: input vector, c: output vector
    double times[NITER];
    double mu = 0.0, sigma = 0.0; //mu: arithmetic mean of execution times, sigma: standard deviation of execution times
    int sizeBlock; //Block size


	#ifdef RAND

    int code = atoi(argv[5]); //Seed for randomization
    sizeBlock = atoi(argv[4]);

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    nonZeros = atoi(argv[3]);

    #else

    char line[1024];

    sizeBlock = atoi(argv[2]);

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

    //Allocate Unified Memory accessible from CPU or GPU
   	#ifdef Managed
    cudaMallocManaged(&b, m*sizeof(dtype));
    cudaMallocManaged(&c, n*sizeof(dtype));
    #endif

    //Allocate Pinned Memory accessible from GPU
    #ifdef Pinned
    cudaMallocHost(&b, m*sizeof(dtype));
    cudaMallocHost(&c, n*sizeof(dtype));
    //cudaMallocHost(&tmp, n*sizeof(dtype));
    #endif

    int *COOr; //Row vector of the COO format matrix
    int *COOc; //Column vector of the COO format matrix
    dtype *COOv; //Value vector of the COO format matrix

    //Allocate Unified Memory accessible from CPU or GPU
    #ifdef Managed
    cudaMallocManaged(&COOr, nonZeros*sizeof(int));
    cudaMallocManaged(&COOc, nonZeros*sizeof(int));
    cudaMallocManaged(&COOv, nonZeros*sizeof(dtype));
    #endif

    //Allocate Pinned Memory accessible from GPU
    #ifdef Pinned
    cudaMallocHost(&COOr, nonZeros*sizeof(int));
    cudaMallocHost(&COOc, nonZeros*sizeof(int));
    cudaMallocHost(&COOv, nonZeros*sizeof(dtype));
    #endif

    //Determinate the block and grid size
	int threads; //Block size
    int blocks; //Grid size

    #ifdef COO3
    if(sizeBlock < 1){
      	if(1024 > n){
        	threads = n;
        	blocks = 1;
    	}else{
        	threads = 1024;
        	blocks = (n + threads - 1) / threads;
    	}
    }else{

    	if(sizeBlock >= 1024){
        	threads = 1024;
        	blocks = (n + threads - 1) / threads;
      	}else{
        	threads = sizeBlock;
        	blocks = (n + threads - 1) / threads;

      	}

    }
    #endif

    #if defined(COO4) || defined(COO1) || defined(COO2) || defined(COO5)

    if(sizeBlock < 1){
      	if(1024 > nonZeros){
        	threads = nonZeros;
        	blocks = 1;
    	}else{
        	threads = 1024;
        	blocks = (nonZeros + threads - 1) / threads;
   	 	}
    }else{

    	if(sizeBlock >= 1024){
        	threads = 1024;
        	blocks = (nonZeros + threads - 1) / threads;
      	}else{
        	threads = sizeBlock;
        	blocks = (nonZeros + threads - 1) / threads;
      	}

    }

    #endif


	int typ = (strcmp( XSTR(dtype) ,"int")==0); //Indicate the type of value: typ=true (int),type=false (others)

    printf("Fill matrix\n");

    #ifdef RAND

    init_matrixI(1, nonZeros, COOr, n);
    init_matrixI(1, nonZeros, COOc, m);
    init_matrixV(1, nonZeros, COOv, 0);

    initialize_random_coo(COOr, COOc, COOv, nonZeros, n, m,code);

    #else

    char* buffer = (char *)malloc(size + 1);
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

    //initialize the matrix
    init_matrixV(1,n,c,0);


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

    #ifdef Pinned
    dtype *d_b, *d_c; //GPU input and output vector
    int *d_COOr; //GPU row vector of the COO format matrix
    int *d_COOc; //GPU column vector of the COO format matrix
    dtype *d_COOv; //GPU value vector of the COO format matrix

    cudaMalloc(&d_b, m*sizeof(dtype));
    cudaMalloc(&d_c, n*sizeof(dtype));
    cudaMalloc(&d_COOr, nonZeros*sizeof(int));
    cudaMalloc(&d_COOc, nonZeros*sizeof(int));
    cudaMalloc(&d_COOv, nonZeros*sizeof(dtype));

    cudaMemcpy(d_b, b, m*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOr, COOr, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOc, COOc, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOv, COOv, nonZeros*sizeof(dtype), cudaMemcpyHostToDevice);
    #endif

    printf("calculate moltiplication\n");

    #ifdef COO1
    multiplicationCOO1<<<blocks, threads>>>(COOc, COOv,b,nonZeros);

   	cudaDeviceSynchronize();

   	int rowr;
   	for(i = 0; i < nonZeros; i++){
        rowr = COOr[i];
     	c[rowr] += COOv[i];
   	}
    #endif

    #ifdef COO3
    //printf("blocks and threads: %d and %d\n", blocks, threads);
    multiplicationCOO3<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros, n);
    #endif

    #ifdef COO2
    multiplicationCOO2<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
    #endif

    #ifdef COO4
	multiplicationCOO4<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
    #endif

    #ifdef COO5
    size_t sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);
	multiplicationCOO5<<<blocks,threads,sharedMemSize>>>(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);
    #endif


    // Wait for GPU to finish before accessing on host
    #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO5)
    cudaDeviceSynchronize();
    #endif

    #ifdef Pinned
    cudaMemcpy(c, d_c, n*sizeof(dtype), cudaMemcpyDeviceToHost);
    #endif

    #ifdef Print
    if(typ){
   	    PRINT_RESULT_VECTORI(c, "C", n);
    }else{
        PRINT_RESULT_VECTORF(c, "C", n);
    }
    #endif

    #ifdef Check
    //If the input matrix has integer value then it is checked the correctness of the chosen SpMV solution with CPU solution
    if(typ){
      	dtype *Cpuc = (dtype *)malloc(n*sizeof(dtype));

    	init_matrixV(1,n,Cpuc,0);

    	multiplicationCOO(COOr, COOc, COOv, b, Cpuc, nonZeros);

    	int correct = 1;
    	for (i = 0; i < n; ++i) {
        	if (abs(Cpuc[i] - c[i])) {
            	correct = 0;
            	break;
        	}
    	}

    	printf("SpMV verification: %s\n", correct ? "SUCCESS" : "FAILURE");

        free(Cpuc);

    }

    #endif

	init_matrixV(1,n,c,0);

    #ifdef Pinned
    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    #endif

    #ifndef COO1
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    float millisec;
    #endif

    for (int k=-2; k<NITER; k++) {
      	#if defined(COO1)
	    TIMER_START(0);
        #endif

        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO5)
        cudaEventRecord(start);
        #endif

        #ifdef COO1
        multiplicationCOO1<<<blocks, threads>>>(COOc, COOv,b,nonZeros);

   		cudaDeviceSynchronize();
		int rowr;
   		for(i = 0; i < nonZeros; i++){
            rowr = COOr[i];
     		c[rowr] += COOv[i];
   		}
        #endif

        #ifdef COO3
        multiplicationCOO3<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros, n);
        #endif

        #ifdef COO2
        multiplicationCOO2<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
        #endif

        #ifdef COO4
        multiplicationCOO4<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
        #endif

        #ifdef COO5
		multiplicationCOO5<<<blocks,threads,sharedMemSize>>>(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);
   	 	#endif

        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO5)
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        #endif

        #ifdef COO1
        TIMER_STOP(0);
        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        #endif

        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO5)
        cudaEventElapsedTime(&millisec, start,stop);
	    double iter_time = millisec / 1.e3;
        #endif

	    if( k >= 0) times[k] = iter_time;
        printf("Iteration %d tooks %lfs\n", k, iter_time);
        init_matrixV(1,n,c,0);

        #ifdef Pinned
        cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
        #endif
    }

    #ifndef COO1
    cudaEventDestroy(start);
	cudaEventDestroy(stop);
	#endif

    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");
    mu = mu_fn_sol(times, NITER);
    sigma = sigma_fn_sol(times, mu, NITER);

    printf("platf,matrix,id,n,m,nonZeros,blockSize,Rand,memCopy,sort,mu,sigma,nflop,nMemAc,AI_O,AI_A,AI,Iperf,flops,effBand,RP\n");
    printf("GPU,");
    #ifdef RAND
    printf("Random,");
    #else
    printf("%s,",argv[1]);
    #endif

    #ifdef COO1
    printf("COO1,");

    unsigned int nflop = 2*nonZeros;

    unsigned int nMemAc = sizeof(dtype)*(6*nonZeros) + sizeof(int)*(2*nonZeros);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*6 + sizeof(int)*2;
    double AI = AI_O/AI_A;

    #endif

    #ifdef COO2
    printf("COO2,");

    int nflop = 2*nonZeros;

    unsigned int nMemAc = sizeof(int)*(4*nonZeros + n - 2) + sizeof(dtype)*(n + 2*nonZeros);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*3 + sizeof(int)*5;
    double AI = AI_O/AI_A;

    #endif

    #ifdef COO3
    printf("COO3,");

    //Number of access of COOr in COO3 solution
    unsigned int AccessCoor = 0;
    for(i=0; i<n; i++){
        char continu = 0;
        for(j=0; j<nonZeros; j++){

            if(COOr[j] == i){

                AccessCoor++;
                continu = 1;

            }else{

                AccessCoor++;
                if(continu){

                    break;

                }
            }
        }
    }

    int nflop = 2*nonZeros;

    unsigned int nMemAc = sizeof(dtype)*(n + 2*nonZeros) + sizeof(int)*(nonZeros + AccessCoor);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*3 + sizeof(int)*2;
    double AI = AI_O/AI_A;

    #endif

    #ifdef COO4
    printf("COO4,");

    int nflop = 2*nonZeros;


    unsigned int nMemAc = sizeof(dtype)*(4*nonZeros) + sizeof(int)*(2*nonZeros);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*4 + sizeof(int)*2;
    double AI = AI_O/AI_A;

    #endif

    #ifdef COO5
    printf("COO5,");

    //Calculate the number of atomicAdd done
    int stopSum;

    for(i=0; i<nonZeros; i++){
        if(i == nonZeros-1){
            stopSum++;
        }else{
            if(COOr[i] != COOr[i+1]){
                stopSum++;
            }else{
                if(i%threads == 0){
                    stopSum++;
                }
            }
        }
    }


    int nflop = 2*nonZeros;

    unsigned int nMemAc = sizeof(dtype)*(2*nonZeros + 2*stopSum) + sizeof(int)*2*nonZeros;

    double AI_O = 2;
    int AI_A = 3*sizeof(dtype) + sizeof(int)*2;
    double AI = AI_O/AI_A;

    #endif


    printf("%d,%d,%d,%d,",n,m,nonZeros,threads);
    #ifdef RAND
    printf("yes,");
    #else
    printf("no,");
    #endif
    #ifdef Pinned
    printf("Pinned,");
    #else
    printf("Managed,");
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

    double Iperf = AI*MB;

    double flops = (nflop / mu)/1.e9;

    double effBand = (nMemAc/1.e9)/mu;

    double RP = 11.039657;
    //double RP = 106.01851852;

    printf("%lf,%lf,%d,%u,%lf,%d,%lf,%lf,%lf,%lf,%lf",mu,sigma,nflop,nMemAc,AI_O,AI_A,AI,Iperf,flops,effBand,RP);

    #ifdef Pinned
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_COOr);
    cudaFree(d_COOc);
    cudaFree(d_COOv);

    cudaFreeHost(COOr);
    cudaFreeHost(COOc);
    cudaFreeHost(COOv);
    cudaFreeHost(b);
    cudaFreeHost(c);
    #endif

    #ifdef Managed
    cudaFree(COOr);
    cudaFree(COOc);
    cudaFree(COOv);
    cudaFree(b);
    cudaFree(c);
    #endif

    #ifndef RAND

    fclose(fp);

    #endif



    return(0);
}

