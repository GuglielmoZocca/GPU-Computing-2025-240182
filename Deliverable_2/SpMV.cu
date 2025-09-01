#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda/pipeline>

#include "include/time_lib.h"
#include "include/utils.h"

#define NITER 10 //number of iterations test

#define dtype float //Type of the data

#define PFP32 10300 //Peak FP32 Compute Performance
#define MB 933 //Peak Memory Bandwidth

#define WARP_SIZE 32 //dimension of the warp

#if !defined(COO_OLD) && !defined(COO_CUSPARSE) && !defined(COO_NEW_1)
#error "The algorithm is not defined (COO, COO_CUSPARSE,COO_NEW_1)"
#endif

#if (defined(COO_OLD) && (defined(COO_CUSPARSE) || defined(COO_NEW_1))) || (defined(COO_CUSPARSE) && (defined(COO_OLD) || defined(COO_NEW_1))) || (defined(COO_NEW_1) && (defined(COO_OLD) || defined(COO_CUSPARSE)))
#error "You can only define COO or COO_CUSPARSE or COO_NEW_1"
#endif

#if defined(SortC) && defined(SortR)
#error "Only a sort can be specified"
#endif

#if defined(COO_CUSPARSE) && !defined(SortR)
#error "You can define COO_CUSPARSE only with SortR"
#endif

//utility function
#define CHECK_CUDA(func)                                                        \
    {                                                                           \
        cudaError_t status = (func);                                            \
        if (status != cudaSuccess) {                                            \
            std::cerr << "CUDA API failed at line " << __LINE__ << ": "        \
                      << cudaGetErrorString(status) << std::endl;              \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    }

//utility function
#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            std::cerr << "cuSPARSE API failed at line " << __LINE__ << ": "    \
                      << status << std::endl;                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }




//CPU SpMV implementation COO
void multiplicationCOOCPU(int *COOR, int *COOC, dtype *COOV, dtype *V, dtype *R, int nonZ){
    int i;
    int row;
    for (i=0; i<nonZ; i++){

            row = COOR[i];
            R[row] += COOV[i]*V[COOC[i]];

    }
}

//Deliverable 1 GPU SpMV Shared Memory Optimized Solution
__global__
void multiplicationCOO_OLD(int *COOR, int *COOC, dtype *COOV,dtype *V, dtype *R, int nnz) {
	extern __shared__ char shared_mem[]; //dynamically allocated shared memory
	int *s_rows = (int*)shared_mem; //shared memory of rows
	dtype *s_vals = (dtype*)&s_rows[blockDim.x]; //shared memory of multiplied values

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int local_tid = threadIdx.x;

	if (tid >= nnz) return;

	s_rows[local_tid] = COOR[tid];
	s_vals[local_tid] = COOV[tid]*V[COOC[tid]];
	__syncthreads();

    //the reduction part is done by the thread at the limit of row and block
	if (local_tid > 0 && s_rows[local_tid] == s_rows[local_tid - 1]) return;

	dtype sum = s_vals[local_tid];
	int row = s_rows[local_tid];

    //accumulate the values until the corripsondent row or block is ended
	for (int i = local_tid + 1; i < blockDim.x && tid + (i - local_tid) < nnz; i++) {
		if (s_rows[i] == row) {
			sum += s_vals[i];
		} else {
			break;
		}
	}

	atomicAdd(&R[row], sum);
}

#ifdef COO_NEW_1

#define MAX_BLOCKS 200

//Kernel to process the last elements in COO_NEW_1 solution (NEW version of the third Kernel)
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_kernel_shared(const int *COOR,const int *COOC,const dtype *COOV,const dtype *V, dtype *R, int nnz) {
	extern __shared__ char shared_mem[]; //dynamically allocated shared memory
	int *s_rows = (int*)shared_mem; //shared memory of rows
	dtype *s_vals = (dtype*)&s_rows[blockDim.x]; //shared memory of multiplied values

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int local_tid = threadIdx.x;

	if (tid >= nnz) return;

	s_rows[local_tid] = __ldg(&COOR[tid]);
	s_vals[local_tid] = __ldg(&COOV[tid])*__ldg(&V[__ldg(&COOC[tid])]);
	__syncthreads();

    //the reduction part is done by the thread at the limit of row and block
	if (local_tid > 0 && s_rows[local_tid] == s_rows[local_tid - 1]) return;

	dtype sum = s_vals[local_tid];
	int row = s_rows[local_tid];

    //accumulate the values until the corripsondent row or block is ended
	for (int i = local_tid + 1; i < blockDim.x && tid + (i - local_tid) < nnz; i++) {
		if (s_rows[i] == row) {
			sum += s_vals[i];
		} else {
			break;
		}
	}

	atomicAdd(&R[row], sum);
}

//Prefix sum of the input elements in a block
__device__ void segreduce_block(const int * idx, dtype * val)
{
  	int row = idx[threadIdx.x];
    dtype left = 0;
    if( threadIdx.x >=   1 && row == idx[threadIdx.x -   1] ) { left = val[threadIdx.x -   1]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   2 && row == idx[threadIdx.x -   2] ) { left = val[threadIdx.x -   2]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   4 && row == idx[threadIdx.x -   4] ) { left = val[threadIdx.x -   4]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   8 && row == idx[threadIdx.x -   8] ) { left = val[threadIdx.x -   8]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  16 && row == idx[threadIdx.x -  16] ) { left = val[threadIdx.x -  16]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  32 && row == idx[threadIdx.x -  32] ) { left = val[threadIdx.x -  32]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  64 && row == idx[threadIdx.x -  64] ) { left = val[threadIdx.x -  64]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 128 && row == idx[threadIdx.x - 128] ) { left = val[threadIdx.x - 128]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 256 && row == idx[threadIdx.x - 256] ) { left = val[threadIdx.x - 256]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 512 && row == idx[threadIdx.x - 512] ) { left = val[threadIdx.x - 512]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
}

//NEW warp-level segmented scan & reduction (First Kernel of the COO_NEW_1)
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_flat_kernel_old_atomic(const int tail,
                     const int interval_size,
                     const int * I,
                     const int * J,
                     const dtype * V,
                     const dtype * x,
                           dtype * y,
                           int * temp_rows,
                           dtype * temp_vals)
{
    __shared__ volatile int rows[48 *(BLOCK_SIZE>>5)]; //shared memory of the rows
    __shared__ volatile dtype vals[BLOCK_SIZE]; //shared memory of the vals


    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;                         // global thread index
    const int thread_lane = threadIdx.x & (WARP_SIZE-1); //threadIdx.x%WARP_SIZE                            // thread index within the warp
    const int warp_id     = thread_id>>5; //thread_id/WARP_SIZE;                                       // global warp index

    const int interval_begin = warp_id * interval_size;                                    // warp's offset into I,J,V
    const int interval_end   = ((interval_begin + interval_size) > tail) ? tail : (interval_begin + interval_size); // end of warps's work

    const int idx = 16 * ((threadIdx.x>>5) + 1) + threadIdx.x;                        // thread's index into padded rows array

    rows[idx - 16] = -1;                                                                         // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                           // warp has no work to do
        return;

   	int row_start = I[0]; //first row considerated in the stream
    int row_end   = I[tail-1]; //last row considerated in the stream

    if (thread_lane == 31)
    {
        // initialize the carry in values
        rows[idx] = I[interval_begin];
		vals[threadIdx.x] = dtype(0);
    }

    //threads in the warp cooperate to process every element in a specific interval
    for(int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        int row = __ldg(&I[n]);                                         // row index (i)
        dtype val = __ldg(&V[n]) * __ldg(&x[__ldg(&J[n])]);            // A(i,j) * x(j)

        if (thread_lane == 0)
        {

          	// row continues
            if(row == rows[idx + 31]){
                val += vals[threadIdx.x + 31];
            }else{// row terminated
              	if(rows[idx + 31] != row_start && rows[idx + 31] != row_end){
                	y[rows[idx + 31]] += vals[threadIdx.x + 31];
				}else{
                    atomicAdd(&y[rows[idx + 31]], vals[threadIdx.x + 31]);

				}
            }
        }

        rows[idx]         = row;
        vals[threadIdx.x] = val;

		//Prefix sum the input elements in the warp
        if(row == rows[idx -  1]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; }
        if(row == rows[idx -  2]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
        if(row == rows[idx -  4]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
        if(row == rows[idx -  8]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
        if(row == rows[idx - 16]) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }


        if(thread_lane < 31 && row != rows[idx + 1]){// row terminated
          if(row != row_start && row != row_end){
                	y[row] += vals[threadIdx.x];
			}else{
                    atomicAdd(&y[row], vals[threadIdx.x]);
			}
        }
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = rows[idx];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}

//BASE warp-level segmented scan & reduction (First Kernel of the COO_NEW_1)
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_flat_kernel_old(const int tail,
                     const int interval_size,
                     const int * I,
                     const int * J,
                     const dtype * V,
                     const dtype * x,
                           dtype * y,
                           int * temp_rows,
                           dtype * temp_vals)
{
    __shared__ volatile int rows[48 *(BLOCK_SIZE>>5)]; //shared memory of the rows
    __shared__ volatile dtype vals[BLOCK_SIZE]; //shared memory of the vals


    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;                         // global thread index
    const int thread_lane = threadIdx.x & (WARP_SIZE-1);                                   // thread index within the warp
    const int warp_id     = thread_id>>5;//thread_id/WARP_SIZE;                                       // global warp index

    const int interval_begin = warp_id * interval_size;                                    // warp's offset into I,J,V
    const int interval_end   = ((interval_begin + interval_size) > tail) ? tail : (interval_begin + interval_size);  // end of warps's work

    const int idx = 16 * ((threadIdx.x>>5) + 1) + threadIdx.x; //  threadIdx.x/32                          // thread's index into padded rows array

    rows[idx - 16] = -1;                                                                         // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                           // warp has no work to do
        return;

    if (thread_lane == 31)
    {
        // initialize the carry in values
        rows[idx] = I[interval_begin];
		vals[threadIdx.x] = dtype(0);
    }

	//threads in the warp cooperate to process every element in a specific interval
    for(int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        int row = I[n];                                         // row index (i)
        dtype val = V[n] * x[J[n]];            // A(i,j) * x(j)

        if (thread_lane == 0)
        {
            // row continues
            if(row == rows[idx + 31]){
                val += vals[threadIdx.x + 31];
            }else{// row terminated
                	y[rows[idx + 31]] += vals[threadIdx.x + 31];

            }

        }

        rows[idx]         = row;
        vals[threadIdx.x] = val;

		//Prefix sum the input elements in the warp
        if(row == rows[idx -  1]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; }
        if(row == rows[idx -  2]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
        if(row == rows[idx -  4]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
        if(row == rows[idx -  8]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
        if(row == rows[idx - 16]) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

        // row terminated
        if(thread_lane < 31 && row != rows[idx + 1]){

                	y[row] += vals[threadIdx.x];


        }
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = rows[idx];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}


//NEW block-level segmented scan and reduction (Second Kernel of the COO_NEW_1)
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_reduce_update_kernel_mult_block(const int num_warps,
                              const int * temp_rows,
                              const dtype * temp_vals,
                                    dtype * y)
{
    __shared__ int rows[BLOCK_SIZE + 1]; //shared memory of the rows
    __shared__ dtype vals[BLOCK_SIZE + 1]; //shared memory of the multiplied values

    const int end = blockIdx.x*BLOCK_SIZE + BLOCK_SIZE;  //id of the ending element in the block

    if (threadIdx.x == 0)
    {
      	//end point of the processing
        rows[BLOCK_SIZE] = (int) -1;
        vals[BLOCK_SIZE] = (dtype)  0;
    }

    __syncthreads();

    int i = blockIdx.x*BLOCK_SIZE + threadIdx.x; //thread id

    //The part of elements considerated fit the entire block
	if (end <= num_warps){

    rows[threadIdx.x] = __ldg(&temp_rows[i]);
    vals[threadIdx.x] = __ldg(&temp_vals[i]);

    __syncthreads();

    //prefix sum of the input elements in a block
    segreduce_block(rows, vals);

    // row terminated
    if (rows[threadIdx.x] != rows[threadIdx.x + 1])
      atomicAdd(&y[rows[threadIdx.x]], vals[threadIdx.x]);



    } else {//The part of elements considerated fit a part of the block
        if (i < num_warps){
            rows[threadIdx.x] = __ldg(&temp_rows[i]);
            vals[threadIdx.x] = __ldg(&temp_vals[i]);
        } else {
          	//end point of the processing
            rows[threadIdx.x] = (int) -1;
            vals[threadIdx.x] = (dtype)  0;
        }

        __syncthreads();

        //prefix sum of the input elements in a block
        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
              atomicAdd(&y[rows[threadIdx.x]], vals[threadIdx.x]);
    }
}

//BASE block-level segmented scan and reduction (Second Kernel of the COO_NEW_1)
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_reduce_update_kernel_one_block(const int num_warps,
                              const int * temp_rows,
                              const dtype * temp_vals,
                                    dtype * y)
{
    __shared__ int rows[BLOCK_SIZE + 1]; //shared memory of the rows
    __shared__ dtype vals[BLOCK_SIZE + 1]; //shared memory of the multiplied values

    const int end = num_warps - (num_warps & (BLOCK_SIZE - 1)); //do not considerate the elements that don't fit in a block

    if (threadIdx.x == 0)
    {
      	//end point of the processing
        rows[BLOCK_SIZE] = (int) -1;
        vals[BLOCK_SIZE] = (dtype)  0;
    }

    __syncthreads();

    int i = threadIdx.x;

    while (i < end)
    {
        rows[threadIdx.x] = temp_rows[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        //prefix sum of the input elements in a block
        segreduce_block(rows, vals);

        // row terminated
        if (rows[threadIdx.x] != rows[threadIdx.x + 1])
            y[rows[threadIdx.x]] += vals[threadIdx.x];

        __syncthreads();

        i += BLOCK_SIZE;
    }

    //process the last elements that don't fit in a block
    if (end < num_warps){
        if (i < num_warps){
            rows[threadIdx.x] = temp_rows[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
          	//end point of the processing
            rows[threadIdx.x] = (int) -1;
            vals[threadIdx.x] = (dtype)  0;
        }

        __syncthreads();

        //prefix sum of the input elements in a block
        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
                y[rows[threadIdx.x]] += vals[threadIdx.x];
    }
}

//Serial kernel to process the last elements in COO_NEW_1 solution (Base version of the third kernel)
__launch_bounds__(1,1)
__global__ void
spmv_coo_serial_kernel(const int num_entries,
                       const int * I,
                       const int * J,
                       const dtype * V,
                       const dtype * x,
                             dtype * y)
{
    for(int n = 0; n < num_entries; n++)
    {

        y[I[n]] += V[n] * x[J[n]];

    }
}

//GPU SpMV COO_NEW Solution
double multiplicationCOO_NEW_1(int *COOR, int *COOC, dtype *COOV, const dtype *X, dtype *R, int nonZ)
{

    const int * I = COOR; //rows inidices
    const int * J = COOC; //column indices
    const dtype * V = COOV; //values vector

    if(nonZ == 0)
    {
        // empty matrix
        return -1;
    }
    else if (nonZ < static_cast<size_t>(WARP_SIZE))
    {
        // small matrix
        spmv_coo_serial_kernel<<<1,1>>>(nonZ, I, J, V, X, R);
        return -1;
    }

    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int part = (nonZ/N_STREAM); //Considerated input matrix for the stream

    const unsigned int num_units  = part / WARP_SIZE; //number of section processed by a warp

    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS); //number of warps

    const unsigned int num_blocks = (num_warps + (WARPS_PER_BLOCK - 1)) / WARPS_PER_BLOCK; //number of processing blocks

    const unsigned int num_iters  = (num_units + (num_warps - 1)) / num_warps; //number of interaction on the matrix by a warp

    const unsigned int interval_size = WARP_SIZE * num_iters; //number of elements considerated by a warp

    const int tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)

    const unsigned int active_warps = (interval_size == 0) ? 0 : ((tail + (interval_size - 1)) / interval_size); //number of processing warps

    int* temp_rows[N_STREAM]; //temporary rows vector
    dtype*   temp_vals[N_STREAM]; //product of the first level of reduction

    unsigned int step = 0; //variable to accumulate the elements processed by a stream

    //stream and temporary vectors instantiation
 	cudaStream_t stream[N_STREAM];
	for(int i = 0; i < N_STREAM; i++){
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        cudaMalloc(&temp_rows[i], active_warps*sizeof(int));
    	cudaMalloc(&temp_vals[i], active_warps*sizeof(dtype));
    }

    #ifndef EVAL_NCU

    //event instantiation
    cudaEvent_t starto0, stopo0;
    float millisec0;
    cudaEventCreate(&starto0);
    cudaEventCreate(&stopo0);
	#endif

    int blocks; //number of block of third kernel
    int threads; //number of threads of third kernel
    size_t sharedMemSize; //shared memory size of the third kernel

    #ifndef EVAL_NCU
    cudaEventRecord(starto0);
    #endif

    //iterate on every streams
    for(int w = 0; w < N_STREAM; w++){

      	//warp-level segmented scan & reduction (First Kernel of the COO_NEW_1)
		if(N_STREAM == 1){ //don't need an atomic with a single stream
    		spmv_coo_flat_kernel_old<<<num_blocks, BLOCK_SIZE,0,stream[w]>>>(tail, interval_size, I + step, J + step, V + step, X, R,temp_rows[w], temp_vals[w]);
        } else {
        	spmv_coo_flat_kernel_old_atomic<<<num_blocks, BLOCK_SIZE,0,stream[w]>>>(tail, interval_size, I + step, J + step, V + step, X, R,temp_rows[w], temp_vals[w]);
        }

        //block-level segmented scan and reduction (Second Kernel of the COO_NEW_1)
        if(N_STREAM == 1){ //don't need an atomic with a single stream
			spmv_coo_reduce_update_kernel_one_block<<<1, BLOCK_SIZE,0,stream[w]>>>(active_warps, temp_rows[w], temp_vals[w], R);
		} else {
            blocks = (active_warps + BLOCK_SIZE - 1) / BLOCK_SIZE;
        	spmv_coo_reduce_update_kernel_mult_block<<<blocks, BLOCK_SIZE,0,stream[w]>>>(active_warps, temp_rows[w], temp_vals[w], R);
		}

		//kernel to process the last elements in COO_NEW_1 solution
        if(w == (N_STREAM-1)){ //the last stream must considerate the missing elements from the division in stream: nonZ - (nonZ/N_STREAM)*N_STREAM
    		if(BLOCK_SIZE > ((part - tail) + (nonZ - (step + part)))){
        		threads = (part - tail) + (nonZ - (step + part));
    			blocks = 1;
    		}else{
        		threads = BLOCK_SIZE;
        		blocks = ((part - tail) + (nonZ - (step + part)) + threads - 1) / threads;
   			}

    		sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);

        	if(N_STREAM == 1){//don't need an atomic with a single stream
        		spmv_coo_serial_kernel<<<1,1>>>((part - tail) + (nonZ - (step + part)),I + tail + step, J + tail + step, V + tail + step, X, R);
            } else {
            	spmv_coo_kernel_shared<<<blocks,threads,sharedMemSize,stream[w]>>>(I + tail + step, J + tail + step, V + tail + step, X, R,(part - tail) + (nonZ - (step + part)));
            }

        }else{
          if(BLOCK_SIZE > (part - tail)){
        		threads = (part - tail);
    			blocks = 1;
    		}else{
        		threads = BLOCK_SIZE;
        		blocks = ((part - tail) + threads - 1) / threads;
   			}

    		sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);


            spmv_coo_kernel_shared<<<blocks,threads,sharedMemSize,stream[w]>>>( I + tail + step, J + tail + step, V + tail + step, X, R,part - tail);
        }

        //next matrix part for the next stream
        step += part;

    }

    for(int i = 0; i < N_STREAM; i++){
          CHECK_CUDA(cudaStreamSynchronize(stream[i]));
    }

    #ifndef EVAL_NCU
    cudaEventRecord(stopo0);

    cudaEventSynchronize(stopo0);

    cudaEventElapsedTime(&millisec0, starto0,stopo0);

    //time of the entire spmv process
    double iter_time = millisec0 / 1.e3;
    #endif

   	for(int i = 0; i < N_STREAM; i++){
      	cudaFree(temp_rows[i]);
    	cudaFree(temp_vals[i]);
        cudaStreamDestroy(stream[i]);
    }

    #ifndef EVAL_NCU
    cudaEventDestroy(starto0);
    cudaEventDestroy(stopo0);
    #endif

    #ifndef EVAL_NCU
	return iter_time;
    #else
	return 0.0;
	#endif

}

#endif

int main(int argc, char* argv[]) {
    #ifdef RAND

  	//Capture input parameters
    #if defined(COO_CUSPARSE) || defined(EVAL_NCU) || defined(COO_NEW_1)
    if (argc < 5) {
    printf("Usage: %s n m nonZero code\n", argv[0]);
    #else
  	if (argc < 6) {
	printf("Usage: %s n m nonZero blockSize code\n", argv[0]);
	#endif

	return(1);
    }

    printf("argv[0] = %s\n", argv[1]);
    printf("argv[1] = %s\n", argv[2]);
    printf("argv[2] = %s\n", argv[3]);
    #if !defined(COO_CUSPARSE) && !defined(EVAL_NCU) && !defined(COO_NEW_1)
    printf("argv[3] = %s\n", argv[4]);
    printf("argv[4] = %s\n", argv[5]);
    #else
    printf("argv[3] = %s\n", argv[4]);
    #endif

  	#else

	#if defined(COO_CUSPARSE) || defined(EVAL_NCU) || defined(COO_NEW_1)
    if (argc < 2) {
	printf("Usage: %s *.mtx\n", argv[0]);
	#else
	if (argc < 3) {
	printf("Usage: %s *.mtx blockSize\n", argv[0]);
	#endif

        return(1);
    }


    printf("argv[0] = %s\n", argv[1]);
    #if !defined(COO_CUSPARSE) && !defined(EVAL_NCU) && !defined(COO_NEW_1)
    printf("argv[1] = %s\n", argv[2]);
	#endif

    FILE *fp;

    char *path = concat("matrix/",argv[1]);

    fp = fopen(path, "r");
    if (fp == NULL) {
        printf("\nError; Cannot open file");
        exit(1);
    }

    free(path);

	#endif

    int i;
    int n,m,nonZeros; //n: number of rows, m: number of columns, nonZeros: number of non-zeros
    dtype *b, *c; //b: input vector, c: output vector
	double times[NITER];
    double mu = 0.0, sigma = 0.0; //mu: arithmetic mean of execution times, sigma: standard deviation of execution times

    int sizeBlock; //Block size


	#ifdef RAND

    #if !defined(COO_CUSPARSE) && !defined(EVAL_NCU) && !defined(COO_NEW_1)
    int code = atoi(argv[5]); //Seed for randomization
    sizeBlock = atoi(argv[4]);
    #else
    int code = atoi(argv[4]); //Seed for randomization
    #endif

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    nonZeros = atoi(argv[3]);

    #else

    #if !defined(COO_CUSPARSE) && !defined(EVAL_NCU) && !defined(COO_NEW_1)
    sizeBlock = atoi(argv[2]);
    #endif

	MM_typecode matcode; //matcode del mtx

	//Read the dimension of the file
	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	mm_read_banner(fp, &matcode);

    //don't process complex matrix
    if(mm_is_complex(matcode)) {

      printf("Cannot process complex matrix\n");
      exit(1);

    }
	get_mtx_dims(fp, &n, &m, &nonZeros); //capture the dimension of the matrix

    #endif

    printf("N: %d,M: %d,NonZero: %d\n", n,m,nonZeros);

    //Allocate Pinned Memory accessible from GPU
    cudaMallocHost(&b, m*sizeof(dtype));
    cudaMallocHost(&c, n*sizeof(dtype));

    int *COOr; //Row vector of the COO format matrix
    int *COOc; //Column vector of the COO format matrix
    dtype *COOv; //Value vector of the COO format matrix

    //Allocate Pinned Memory accessible from GPU
    cudaMallocHost(&COOr, nonZeros*sizeof(int));
    cudaMallocHost(&COOc, nonZeros*sizeof(int));
    cudaMallocHost(&COOv, nonZeros*sizeof(dtype));

    //Determinate the block and grid size
	int threads; //Block size
    int blocks; //Grid size
    size_t sharedMemSize; //shared memory size of the COO_OLD solution

    #if defined(COO_OLD)
	#ifndef EVAL_NCU
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

    #endif

    printf("Fill matrix\n");

    #ifdef RAND

    init_matrixI(1, nonZeros, COOr, n);
    init_matrixI(1, nonZeros, COOc, m);
    init_matrixV<dtype>(1, nonZeros, COOv, 0);
    //initialize the matrix with random values
    initialize_random_coo<dtype>(COOr, COOc, COOv, nonZeros, n, m,code);

    #else

    //initialize the different COO structures with the file data
	int ret_code = mtx_to_COO(fp,COOr,COOc,COOv,nonZeros,size,matcode);
    if(ret_code == 1){
    	return 1;
    }
    #endif

    #ifdef SortR
    printf("SortR\n");
    sort_cooR<dtype>(COOr,COOc,COOv,nonZeros); //sort the matrix by row
    #endif
    #ifdef SortC
    printf("SortC\n");
    sort_cooC<dtype>(COOr,COOc,COOv,nonZeros);//sort the matrix by column
    #endif

    //initialize the matrix
    init_matrixV<dtype>(1,n,c,0);

    //initialiaze the input vector
    #ifndef RAND
    if (mm_is_integer(matcode)) {
        for (i=0; i<m; i++) {
            b[i] = 1;
        }
    } else {
        for (i=0; i<m; i++) {
            b[i] = 1.0;
        }
    }
	#else

    for (i=0; i<m; i++) {
        b[i] = 1;
    }

    #endif

    dtype *d_b, *d_c; //GPU input and output vector
    cudaMalloc(&d_b, m*sizeof(dtype));
    cudaMalloc(&d_c, n*sizeof(dtype));
    cudaMemcpy(d_b, b, m*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    #if defined(COO_OLD) || defined(COO_CUSPARSE) || defined(COO_NEW_1)
    int *d_COOr; //GPU row vector of the COO format matrix
    int *d_COOc; //GPU column vector of the COO format matrix
    dtype *d_COOv; //GPU value vector of the COO format matrix

    cudaMalloc(&d_COOr, nonZeros*sizeof(int));
    cudaMalloc(&d_COOc, nonZeros*sizeof(int));
    cudaMalloc(&d_COOv, nonZeros*sizeof(dtype));

    cudaMemcpy(d_COOr, COOr, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOc, COOc, nonZeros*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COOv, COOv, nonZeros*sizeof(dtype), cudaMemcpyHostToDevice);
    #endif

    #if defined(COO_CUSPARSE)

    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
    // cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create matrix descriptor (COO-like parameters)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CHECK_CUSPARSE(cusparseCreateCoo(&matA, n, m, nonZeros,
                                     d_COOr, d_COOc, d_COOv,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, m, d_b, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_c, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;

    // SpMV buffer size and workspace
    size_t bufferSize = 0;
    size_t *dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
	#endif

    #ifndef EVAL_NCU

    printf("calculate moltiplication\n");

    #if defined(COO_CUSPARSE)
    // Execute CUSPARSE SpMV
    CHECK_CUSPARSE(cusparseSpMV(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    #endif

    #if defined(COO_NEW_1)

    // Execute COO_NEW_1 SpMV
    multiplicationCOO_NEW_1(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);

    #endif


    #ifdef COO_OLD

    // Execute COO_OLD SpMV
    sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);
	multiplicationCOO_OLD<<<blocks,threads,sharedMemSize>>>(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);

    #endif

    #if defined(COO_OLD) || defined(COO_NEW_1)

    cudaDeviceSynchronize();

    #endif

    cudaMemcpy(c, d_c, n*sizeof(dtype), cudaMemcpyDeviceToHost);

    #ifdef Print

        PRINT_RESULT_VECTORF(c, "C", n);

    #endif

    #ifdef Check
    #ifndef RAND

    //If the input matrix has integer value then it is checked the correctness of the chosen SpMV solution with CPU solution
    if(mm_is_integer(matcode)) {

    #endif

      	dtype *Cpuc = (dtype *)malloc(n*sizeof(dtype));

    	init_matrixV<dtype>(1,n,Cpuc,0);

    	multiplicationCOOCPU(COOr, COOc, COOv, b, Cpuc, nonZeros);

    	int correct = 1;
    	for (i = 0; i < n; ++i) {
        	if (abs(Cpuc[i] - c[i])) {
            	correct = 0;
            	break;
        	}
    	}

    	printf("SpMV verification: %s\n", correct ? "SUCCESS" : "FAILURE");

        free(Cpuc);
	#ifndef RAND
    }
    #endif
    #endif

	init_matrixV<dtype>(1,n,c,0);

    cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);

    #endif

	#if !defined(EVAL_NCU)
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    float millisec;
    #endif

    #if defined(EVAL_NCU) && !defined(COO_CUSPARSE) && !defined(COO_NEW_1)
    //considerate multiple block dimensions
    for (i=32; i <= 1024; i=i*2 ){
    if(i > nonZeros){
        threads = nonZeros;
    	blocks = 1;
    }else{
        threads = i;
        blocks = (nonZeros + threads - 1) / threads;
   	}
    #endif

    #if defined(COO_OLD)
    sharedMemSize = threads * sizeof(int) + threads * sizeof(dtype);
    #endif

    //multiple cycle to evaluate the solution
    #if defined(EVAL_NCU)
    for (int k=0; k<3; k++) {
    #else
    for (int k=-2; k<NITER; k++) {
    #endif
 		#if !defined(EVAL_NCU)
        cudaEventRecord(start);
        #endif

        #if defined(COO_CUSPARSE)
        #if defined(SortR) && defined(EVAL_NCU)
        char str[9];
        sprintf(str,"%d %d %d", 1,1,k);
        nvtxRangePushA(concat(concat(concat("Cusparse SortR ",argv[1])," "),str));
        #endif
        #if defined(SortC) && defined(EVAL_NCU)
        char str[9];
        sprintf(str,"%d %d %d", 1,1,k);
        nvtxRangePushA(concat(concat(concat("Cusparse SortC ",argv[1])," "),str));
        #endif
        // Execute CUSPARSE SpMV
    	CHECK_CUSPARSE(cusparseSpMV(
    	handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    	&alpha, matA, vecX, &beta, vecY,
    	CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        #if defined(EVAL_NCU)
        nvtxRangePop();
        #endif

    	#endif

        #if defined(COO_OLD)
		#if defined(SortR) && defined(EVAL_NCU)
        char str[9];
        sprintf(str,"%d %d %d", 1,threads,k);
        nvtxRangePushA(concat(concat(concat("COO_OLD SortR ",argv[1])," "),str));
        #endif
        #if defined(SortC) && defined(EVAL_NCU)
        char str[9];
        sprintf(str,"%d %d %d", 1,threads,k);
        nvtxRangePushA(concat(concat(concat("COO_OLD SortC ",argv[1])," "),str));
        #endif
        // Execute COO_OLD SpMV
		multiplicationCOO_OLD<<<blocks,threads,sharedMemSize>>>(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);
        #if defined(EVAL_NCU)
        nvtxRangePop();
        #endif
		#endif

        #if defined(COO_NEW_1)
		#if defined(EVAL_NCU)
        char str[9];
        sprintf(str,"%d %d %d", N_STREAM,BLOCK_SIZE,k);
        nvtxRangePushA(concat(concat(concat("COO_NEW_1 SortR ",argv[1])," "),str));
        #endif
		// Execute COO_NEW_1 SpMV
		double i_time = multiplicationCOO_NEW_1(d_COOr,d_COOc, d_COOv, d_b, d_c, nonZeros);
        #if defined(EVAL_NCU)
        nvtxRangePop();
        #endif
        #endif

        #if !defined(EVAL_NCU)
        #if !defined(COO_NEW_1)
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millisec, start,stop);
        //time of the entire process
	    double iter_time = millisec / 1.e3;
        #else
        double iter_time = i_time;
        #endif

	    if( k >= 0) times[k] = iter_time;
        printf("Iteration %d tooks %f\n", k, iter_time);
        #endif
        init_matrixV<dtype>(1,n,c,0);

        cudaMemcpy(d_c, c, n*sizeof(dtype), cudaMemcpyHostToDevice);
    }

    #if defined(EVAL_NCU) && !defined(COO_CUSPARSE) && !defined(COO_NEW_1)
    }
    #endif

    #if !defined(EVAL_NCU)  && !defined(COO_NEW_1)
    cudaEventDestroy(start);
	cudaEventDestroy(stop);
    #endif

    #ifdef Eval
    printf( "%d iterations performed\n\n", NITER);

    printf("calculate performance\n");
    mu = geometricMean(times, NITER); //geometric mean of the execution times
    sigma = geometricStandardDeviation(times, mu, NITER); //standard deviation of the execution times

    //print test data
    printf("matrix,Msparsity,Mtype,Mpattern,id,n,m,nonZeros,blockSize,Rand,sort,mu,sigma,nflop,nMemAcGL,nMemAcSH,AI_O,AI_A,AI,Iperf,flops,effBandGL,effBandSH,RP,N_STREAM\n");
    #ifdef RAND
    printf("Random,NULL,NULL,NULL,");
    #else
    printf("%s,",argv[1]);
    printf("%c,",matcode[1]);
    printf("%c,",matcode[2]);
    printf("%c,",matcode[3]);


    #endif

    #ifdef COO_OLD
    printf("COO_OLD,");

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


    int nflop = 2*nonZeros; //number of flop operation

    unsigned int nMemAcGL = sizeof(dtype)*(2*nonZeros + 2*stopSum) + sizeof(int)*2*nonZeros; //number of global memory access
    unsigned int nMemAcSH = sizeof(dtype)*(2*nonZeros) + sizeof(int)*2*nonZeros; //number of shared memory access

    double AI_O = 2;
    int AI_A = 3*sizeof(dtype) + sizeof(int)*2;
    double AI = AI_O/AI_A; //Arithmetic Intensity

    #endif

    #ifdef COO_CUSPARSE
    printf("COO_CUSPARSE,");

    int nflop = 2*nonZeros;

    #endif

    #ifdef COO_NEW_1
    printf("COO_NEW_1,");

    int nflop = 2*nonZeros;

    #endif

    #if !defined(COO_CUSPARSE)
    #if defined(COO_NEW_1)
    printf("%d,%d,%d,%d,",n,m,nonZeros,BLOCK_SIZE);
    #else
    printf("%d,%d,%d,%d,",n,m,nonZeros,threads);
	#endif
    #else
	printf("%d,%d,%d,%d,",n,m,nonZeros,-1);
    #endif
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

    #ifdef COO_OLD
    double Iperf = AI*MB; //ideal performance
    double effBandGL = (nMemAcGL/1.e9)/mu; //effective global memory bandwidth
    double effBandSH = (nMemAcSH/1.e9)/mu; //effective shared memory bandwidth
    #endif

    double flops = (nflop / mu)/1.e9; //flop operation for second
    double RP = 11.039657; //Roofline peak

    #ifndef COO_CUSPARSE
    #ifdef COO_NEW_1
    printf("%lf,%lf,%d,%u,%u,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%d\n",mu,sigma,nflop,0,0,0.0,0,0.0,0.0,flops,0.0,0.0,RP,N_STREAM);
    #else
    printf("%lf,%lf,%d,%u,%u,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%d\n",mu,sigma,nflop,nMemAcGL,nMemAcSH,AI_O,AI_A,AI,Iperf,flops,effBandGL,effBandSH,RP,1);
    #endif
	#else
	printf("%lf,%lf,%d,%u,%u,%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%d\n",mu,sigma,nflop,0,0,0.0,0,0.0,0.0,flops,0.0,0.0,RP,1);
    #endif
    #endif

	//free every vector
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFree(d_COOr);
    cudaFree(d_COOc);
    cudaFree(d_COOv);

    cudaFreeHost(COOr);
    cudaFreeHost(COOc);
    cudaFreeHost(COOv);

    #if defined(COO_CUSPARSE)
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    #endif


    cudaFreeHost(b);
    cudaFreeHost(c);

    #ifndef RAND

    fclose(fp);

    #endif



    return(0);
}