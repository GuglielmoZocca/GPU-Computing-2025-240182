#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define dtype int
#define NITER 10
//#define RAND
#define COO5
#define Pinned
#define SortR
//#define Print

#define PFP32 91.6
#define MB 864
#define RP PFP32/MB



#include "include/my_time_lib.h"

typedef struct {
    int row;
    int col;
    double val;
} COOTuple;

__global__
void multiplicationCOO1(int *COOR, int *COOC, dtype *COOV, dtype *V, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    //int j = blockIdx.y*blockDim.y+threadIdx.y;

    if (i < nonZ){
            //if(COOR[j] == i){

                COOV[i] = COOV[i]*V[COOC[i]];

            //}
    }

}

__global__
void multiplicationCOO1_2(int *COOR, dtype *COOV, dtype *V, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    //int j = blockIdx.y*blockDim.y+threadIdx.y;

    if (i < nonZ){
            //if(COOR[j] == i){

                COOV[i] = COOV[i]*V[i];

            //}
    }

}

/*void multiplicationCOO1(dtype *COOR, dtype *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){
   //dim3 threads(1024, 1024);
   //dim3 blocks((N + threads.x - 1) / threads.x,(nonZ + threads.y - 1) / threads.y);
   int blocks;
   int threads;
   if(1024 > nonZ){
        threads = N;
        blocks = 1;
    }else{
       threads = 1024;
        blocks = (nonZ + threads - 1) / threads;
    }

   multiplicationCOO1_Cuda<<<blocks, threads>>>(COOR, COOC, COOV,V,nonZ, N);

   cudaDeviceSynchronize();

   int row = 0;
   int i;
   for(i = 0; i < nonZ; i++){
     R[COOR[i]] += COOV[i];
   }

}*/

/*__global__
void reduce_cuda( dtype *COOR, dtype *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N ) {
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dtype prod = 0;
 	if (idx < nonZ){
        //if(COOR[j] == i){

            COOV[idx] = COOV[idx]*V[COOC[idx]];

        //}
	}else{
          return;
    }
	// convert global pointer to the local pointer of this block
	int * idata = COOV + blockIdx.x * blockDim.x;
	// unrolling 2
	//if ( idx + blockDim.x < n ) g_idata[idx] += g_idata[idx + blockDim.x];
	__syncthreads( );
	// in-place reduction in global memory
	for ( int stride = blockDim.x / 2; stride > 0; stride >>= 1 ) {
		if ( tid < stride ) {
			COOV[ tid ] += COOV[ tid + stride ];
		}
		// synchronize within threadblock
		__syncthreads( );
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

void multiplicationCOO2(dtype *COOR, dtype *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){

    int i = 0;
    int j;
    int row = COOR[i];
    int begin = 0;
	int threads;
    int blocks;
	if(1024 > nonZ){
        threads = nonZ;
        blocks = 1;
    }else{
        threads = 1024;
        blocks = (nonZ + threads - 1) / threads;
    }
    dtype *redu;
    cudaMallocManaged(&redu, blocks*N*sizeof(dtype));
    reduce_cuda( dtype *COOR, dtype *COOC, dtype *COOV, dtype *V, dtype *redu, int nonZ, int N )


}*/

/*__global__ void sumRowsOptimized(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float partialSum[TPB];
    int index = row * ncols + tid;

    float localSum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        localSum += matrix[row * ncols + i];
    }

    partialSum[tid] = localSum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        rowSums[row] = partialSum[0];
    }
}*/

/*__global__
void multiplicationCOO2(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    //printf("ind = %d\n", i);

    if (i < nonZ){

      	extern __shared__ dtype VAL[];

      	VAL[threadIdx.x] = COOV[i]*V[COOC[i]];

        __syncthreads();
  		if(i == 0 || (COOR[i - 1] != COOR[i])){
            int j;
    		dtype prod = 0;
            char continu = 0;
    		int row = COOR[i];
            int idth = threadIdx.x;
    		for(j=i; j<nonZ; j++){

        		if(COOR[j] == row){
					if(idth < blockDim.x){
                      prod += VAL[idth];
                      idth++;
					}else{
                      prod += COOV[j]*V[COOC[j]];
                    }

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
}*/

__global__
void multiplicationCOO2(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;



    //printf("ind = %d\n", i);

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

__global__
void multiplicationCOO3(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ, int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;


    if (i < N){
		dtype prod = 0;
		//printf("ind = %d\n", i);
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

__global__
void multiplicationCOO4(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < nonZ) {
        atomicAdd(&R[COOR[i]], COOV[i]*V[COOC[i]]);
    }
}

__global__
void multiplicationCOO4_2(int *COOR, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < nonZ) {
        atomicAdd(&R[COOR[i]], COOV[i]*V[i]);
    }
}

__global__
void multiplicationCOO5(int *COOR, dtype *COOV,dtype *V, dtype *R, int nnz) {
  	extern __shared__ char shared_mem[]; // Raw pointer
    int *s_rows = (int*)shared_mem; // First half: row indices
    dtype *s_vals = (dtype*)&s_rows[blockDim.x]; // Second half: values

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    if (tid >= nnz) return;

    // Load into shared memory
    s_rows[local_tid] = COOR[tid];
    s_vals[local_tid] = COOV[tid]*V[tid];
    __syncthreads();

    // Each thread reduces values within shared memory for the same row
    if (local_tid > 0 && s_rows[local_tid] == s_rows[local_tid - 1]) return;

    dtype sum = s_vals[local_tid];
    int row = s_rows[local_tid];

    // Reduce next values for the same row
    for (int i = local_tid + 1; i < blockDim.x && tid + (i - local_tid) < nnz; i++) {
        if (s_rows[i] == row) {
            sum += s_vals[i];
        } else {
            break;
        }
    }

    atomicAdd(&R[row], sum);
}


__global__
void multiplicationCSR1(int *CSRR, int *CSRC, dtype *CSRV, dtype *V, dtype *R, int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int n;
    int j;
    int r_s;
    int c_s;
    //for (i=0; i<N; i++){
    if (i < N){

      	dtype mol = 0;

        r_s = CSRR[i];
        c_s = CSRR[i + 1];

        for (j=r_s; j<c_s; j++){

            mol += CSRV[j]*V[CSRC[j]];

        }

        R[i] = mol;
	}
    //}
}

void init_matrixV(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

void init_matrixI(int rows, int cols, int *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*rows +j] = val;
        }
    }
}

void create_CSR(int *CSRRF,int *CSRR, int *CSRC, dtype *CSRV,int N,int nonZ) {

    int i;
    int j;


    init_matrixI(1, N+1, CSRR, 0);

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

void Sort_COO(int *COOR,int *COOC, dtype *COOV, int nonZ){
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


}

int compare_cooR(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->row != ib->row)
        return ia->row - ib->row;
    return ia->col - ib->col;
}

int compare_cooC(const void *a, const void *b) {
    const COOTuple *ia = (const COOTuple *)a;
    const COOTuple *ib = (const COOTuple *)b;

    if (ia->col != ib->col)
        return ia->col - ib->col;
    return ia->row - ib->row;
}

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

void trasform_b(dtype* VO, dtype* VN,int* COOc, int nnz){
  	int i;
	for(i = 0; i < nnz; i++){
  		VN[i] = VO[COOc[i]];
	}
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
    dtype *b, *c, *tmp;
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

    // Allocate Unified Memory accessible from CPU or GPU
   	#ifdef Managed
    cudaMallocManaged(&b, m*sizeof(dtype));
    cudaMallocManaged(&c, n*sizeof(dtype));
    cudaMallocManaged(&tmp, n*sizeof(dtype));
    #endif

    #ifdef Pinned
    cudaMallocHost(&b, m*sizeof(dtype));
    cudaMallocHost(&c, n*sizeof(dtype));
    //cudaMallocHost(&tmp, n*sizeof(dtype));
    #endif

    //COO format
    //#ifdef COO

    int *COOr;
    int *COOc;
    dtype *COOv;

    #ifdef Managed
    cudaMallocManaged(&COOr, nonZeros*sizeof(int));
    cudaMallocManaged(&COOc, nonZeros*sizeof(int));
    cudaMallocManaged(&COOv, nonZeros*sizeof(dtype));
    #endif

    #ifdef Pinned
    cudaMallocHost(&COOr, nonZeros*sizeof(int));
    cudaMallocHost(&COOc, nonZeros*sizeof(int));
    cudaMallocHost(&COOv, nonZeros*sizeof(dtype));
    #endif

    //if(1024 > n && 1024 > nonZeros){
        //dim3 threads(n, nonZeros);
    	//dim3 blocks(1,1);
    //}else{
        //dim3 threads(1024, 1024);
    	//dim3 blocks((n + threads.x - 1) / threads.x,(nonZeros + threads.y - 1) / threads.y);
    //}
	int threads;
    int blocks;

    #ifdef COO3
    if(1024 > n){
        threads = n;
        blocks = 1;
    }else{
        threads = 1024;
        blocks = (n + threads - 1) / threads;
    }
    #endif

    #if defined(COO4) || defined(COO1) || defined(COO2) || defined(COO4_2) || defined(COO1_2) || defined(COO5)

    if(1024 > nonZeros){
        threads = nonZeros;
        blocks = 1;
    }else{
        threads = 1024;
        blocks = (nonZeros + threads - 1) / threads;
    }
    #endif


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
    srand(time(NULL));

    int continua = 1;

    //printf("MAXR: %d\n",RAND_MAX);

    while (numero < nonZeros) {
      	continua = 1;
		r = rand()%n;
        cl = rand()%m;
        for (int l = 0; l < numero; l++) {
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
        //srand(time(NULL));

    }*/







    #else

    //fill the matrix
    int row,col,val;

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    char* buffer = (char *)malloc(size + 1);  // +1 for null terminator
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
    /*if (typ) {
        for(i=0;i<nonZeros;i++) {
      		//printf("%d \n",i);
        	fscanf(fp,"%d %d %d\n", &row,&col,&val);
        	COOr[i]=row-1;
        	COOc[i]=col-1;
        	COOv[i]=val;
    	}
    } else {
        for(i=0;i<nonZeros;i++) {
      		//printf("%d \n",i);
        	fscanf(fp,"%d %d %lf\n", &row,&col,&val);
        	COOr[i]=row-1;
        	COOc[i]=col-1;
        	COOv[i]=val;
   	 	}
    }*/

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

    #if defined(COO4_2) || defined(COO5)
    printf("Trasform b\n");
    dtype* b_2;
    cudaMallocHost(&b_2, nonZeros*sizeof(int));
    trasform_b(b, b_2,COOc, nonZeros);
    #endif
    #ifdef COO1_2
    printf("Trasform b\n");
    dtype* b_2;
    cudaMallocManaged(&b_2, nonZeros*sizeof(int));
    trasform_b(b, b_2,COOc, nonZeros);
    #endif

    #ifdef Pinned
    dtype *d_b, *d_c;
    int *d_COOr;
    int *d_COOc;
    dtype *d_COOv;

    #if defined(COO4_2) || defined(COO5)
    cudaMalloc(&d_b, nonZeros*sizeof(dtype));
    #else
    cudaMalloc(&d_b, m*sizeof(dtype));
    #endif
    cudaMalloc(&d_c, n*sizeof(dtype));
    //cudaMalloc(&d_tmp, nonZeros*sizeof(dtype));
    cudaMalloc(&d_COOr, nonZeros*sizeof(int));
    cudaMalloc(&d_COOc, nonZeros*sizeof(int));
    cudaMalloc(&d_COOv, nonZeros*sizeof(dtype));

    #if defined(COO4_2) || defined(COO5)
    cudaMemcpy(d_b, b_2, nonZeros*sizeof(dtype), cudaMemcpyHostToDevice);
    //printf("COO4_2\n");
    #else
    cudaMemcpy(d_b, b, m*sizeof(dtype), cudaMemcpyHostToDevice);
    //printf("OTHER\n");
    #endif
    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_tmp, d_b, n*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOr, COOr, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOc, COOc, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOv, COOv, nonZeros*sizeof(dtype), cudaMemcpyHostToDevice);
    #endif




    printf("calculate moltiplication\n");

    #ifdef COO1
    multiplicationCOO1<<<blocks, threads>>>(COOr, COOc, COOv,b,nonZeros);

   	cudaDeviceSynchronize();

   	for(i = 0; i < nonZeros; i++){
     	c[COOr[i]] += COOv[i];
   	}
    #endif
    #ifdef COO1_2
    multiplicationCOO1_2<<<blocks, threads>>>(COOr,COOv,b_2,nonZeros);

   	cudaDeviceSynchronize();

   	for(i = 0; i < nonZeros; i++){
     	c[COOr[i]] += COOv[i];
   	}
    #endif
    #ifdef COO3
    printf("blocks and threads: %d and %d\n", blocks, threads);
    multiplicationCOO3<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros, n);
    #endif
    #ifdef COO4
	multiplicationCOO4<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
    #endif
    #ifdef COO4_2
	multiplicationCOO4_2<<<blocks,threads>>>(d_COOr, d_COOv, d_b, d_c, nonZeros);
    #endif
    #ifdef COO5
    size_t sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);
	multiplicationCOO5<<<blocks,threads,sharedMemSize>>>(d_COOr, d_COOv, d_b, d_c, nonZeros);
    #endif
    #ifdef COO2
    //int sharsize = threads * sizeof(dtype);
    //multiplicationCOO2<<<blocks,threads,sharsize>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
    multiplicationCOO2<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
    #endif

    // Wait for GPU to finish before accessing on host
    #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO4_2) || defined(COO5)
    cudaDeviceSynchronize();
    #endif

    //PRINT_RESULT_VECTOR(COOr, "COOr", nonZeros);

    //PRINT_RESULT_VECTOR(COOc, "COOc", nonZeros);

    //PRINT_RESULT_VECTOR(COOv, "COOv", nonZeros);

    //PRINT_RESULT_VECTOR(b, "B", m);

    #ifdef Pinned
    cudaMemcpy(c, d_c, n*sizeof(dtype), cudaMemcpyDeviceToHost);
    #endif

    #ifdef Print
   	PRINT_RESULT_VECTOR(c, "C", n);
    #endif

	init_matrixV(1,n,c,0);

    #ifdef Pinned
    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    #endif

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    float millisec;

    //calculate the moltiplication
    for (int k=-2; k<NITER; k++) {
      	#if defined(COO1) || defined(COO1_2)
	    TIMER_START(0);
        #endif
        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO4_2) || defined(COO5)
        cudaEventRecord(start);
        #endif
        #ifdef COO1
        multiplicationCOO1<<<blocks, threads>>>(COOr, COOc, COOv,b,nonZeros);

   		cudaDeviceSynchronize();

   		for(i = 0; i < nonZeros; i++){
     		c[COOr[i]] += COOv[i];
   		}
        #endif
        #ifdef COO1_2
        multiplicationCOO1_2<<<blocks, threads>>>(COOr,COOv,b_2,nonZeros);

   		cudaDeviceSynchronize();

   		/*for(i = 0; i < nonZeros; i++){
     		c[COOr[i]] += COOv[i];
   		}*/
        #endif
        #ifdef COO3
        multiplicationCOO3<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros, n);
        #endif
        #ifdef COO2
        //multiplicationCOO2<<<blocks,threads,sharsize>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
        multiplicationCOO2<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
        #endif
        #ifdef COO4
        multiplicationCOO4<<<blocks,threads>>>(d_COOr, d_COOc, d_COOv, d_b, d_c, nonZeros);
        #endif
        #ifdef COO4_2
		multiplicationCOO4_2<<<blocks,threads>>>(d_COOr, d_COOv, d_b, d_c, nonZeros);
   	 	#endif
        #ifdef COO5
		multiplicationCOO5<<<blocks,threads,sharedMemSize>>>(d_COOr, d_COOv, d_b, d_c, nonZeros);
   	 	#endif
        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO4_2) || defined(COO5)
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        #endif
        //multiplicationCOO4<<<blocks,threads>>>(COOr, COOc, COOv, b, c, nonZeros, n);
        //cudaDeviceSynchronize();
        #if defined(COO1) || defined(COO1_2)
        TIMER_STOP(0);
        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        #endif
        #if defined(COO4) || defined(COO3) || defined(COO2) || defined(COO4_2) || defined(COO5)
        cudaEventElapsedTime(&millisec, start,stop);
	    double iter_time = millisec / 1.e3;
        #endif
        //double iter_time = TIMER_ELAPSED(0) / 1.e6;
	    if( k >= 0) times[k] = iter_time;

        printf("Iteration %d tooks %lfs\n", k, iter_time);
        init_matrixV(1,n,c,0);
        #ifdef Pinned
        cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
        #endif
    }

    cudaEventDestroy(start);
	cudaEventDestroy(stop);

    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");
    //calculate the performance
    mu = mu_fn_sol(times, NITER);
    sigma = sigma_fn_sol(times, mu, NITER);

    printf(" %10s | %10s | %10s |\n", "v name", "mu(v)", "sigma(v)");
    printf(" %10s | %10f | %10f |\n", "time", mu, sigma);

    #ifdef COO1
    printf("OP: COO1\n");
    unsigned int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(6*nonZeros) + sizeof(int)*(3*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*4 + sizeof(int)*2;
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);

    #endif
    #ifdef COO1_2
    printf("OP: COO1_2\n");
    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(6*nonZeros) + sizeof(int)*(2*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*4 + sizeof(int);
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif
    #ifdef COO3
    printf("OP: COO3\n");

    unsigned int* rowElems = (unsigned int*)malloc(n*sizeof(unsigned int));

    for(int i=0; i<n; i++){
      rowElems[i] = 0;
    }

    for(i=0; i<nonZeros; i++){
      rowElems[COOr[i]]++;
    }

    unsigned int AccessCoor = 0;
    unsigned int AccessCoorTmp = 0;
    for(i=0; i<n; i++){
      AccessCoorTmp = AccessCoor;
      for(j=0; j<rowElems[i]; j++){
        AccessCoor++;
      }
      if(i<n-1){
        AccessCoor++;
      }
      AccessCoor += AccessCoorTmp;
    }

    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*n*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(n + 2*nonZeros) + sizeof(int)*(nonZeros + AccessCoor);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*2 + sizeof(int);
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif
    #ifdef COO4
    printf("OP: COO4\n");
    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(4*nonZeros) + sizeof(int)*(3*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*3 + sizeof(int)*2;
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif
    #ifdef COO4_2
    printf("OP: COO4_2\n");
    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(4*nonZeros) + sizeof(int)*(2*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*3 + sizeof(int);
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif
    #ifdef COO5
    printf("OP: COO5\n");
    int stopSum;

    for(i=0; i<nonZeros; i++){
      if(i != 0 || i == nonZeros-1){
        if(i != nonZeros-1){
          stopSum++;
        }else{
          if(COOr[i] != COOr[i+1]){
            stopSum++;
          }else{
            if(i%(threads-1) == 0){
              stopSum++;
            }
          }
        }
      }
    }


    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(dtype)*(2*nonZeros + stopSum) + sizeof(int)*nonZeros;
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype);
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif
    #ifdef COO2
    int nflop = 2*nonZeros;
    printf("\nMatrix-vector moltiplication COO required 2*nonZeros = %d floating point operations.\n", nflop);

    unsigned int nMemAc = sizeof(int)*(4*nonZeros + 2*n - 2) + sizeof(dtype)*(n + 2*nonZeros);
    printf("\nMatrix-vector moltiplication COO read and write %u bytes.\n", nMemAc);

    double AI_O = 2;
    int AI_A = sizeof(dtype)*2 + sizeof(int);
    double AI = AI_O/AI_A;
    printf("\nAI: %lf\n", AI);
    #endif

    double Iperf = AI*MB;
    printf("\nIdeal performance: %lf\n", Iperf);

    double flops = nflop / mu;
    printf("Matrix-vector moltiplication COO achieved %lf GFLOP/s\n", flops/1.e9);

    double effBand = (nMemAc/1.e9)/mu;
    printf("Matrix-vector moltiplication COO effective bandwidth is %lf GB/s\n", effBand);

    #ifdef Pinned
    cudaFree(d_b);
    cudaFree(d_c);
    //cudaFree(d_tmp);
    cudaFree(d_COOr);
    cudaFree(d_COOc);
    cudaFree(d_COOv);
    #endif
    //free(COOr);
    //free(COOc);
    //free(COOv);
    //free(b);
    //free(c);





    //#endif

    //CSR format
    /*#ifdef CSR

    dtype *CSRr;
    dtype *CSRrF;
    dtype *CSRc;
    dtype *CSRv;

    cudaMallocManaged(&CSRr, (n+1)*sizeof(float));
    cudaMallocManaged(&CSRrF, nonZeros*sizeof(float));
    cudaMallocManaged(&CSRc, nonZeros*sizeof(float));
    cudaMallocManaged(&CSRv, nonZeros*sizeof(float));



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

    multiplicationCSR<<<nbl,ntr>>>(CSRr,CSRc, CSRv, b, c, n);

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
        multiplicationCSR<<<nbl,ntr>>>(CSRr,CSRc, CSRv, b, c, n);
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



    #endif*/

    #ifndef RAND

    fclose(fp);

    #endif



    return(0);
}

