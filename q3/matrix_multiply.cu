#include <stdio.h>
#include <time.h> 	// for clock_gettime( )
#include <errno.h>	// for perror( )
#include <getopt.h>	// for getopt( )
#include <ctype.h>	//for isprint( )


#define GET_TIME(x); if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
						{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }


#ifdef DP
	typedef double real_t;
#else
	typedef float real_t;
#endif
	
#define min(x,y) (x<y?x:y)

#ifndef N
#define N 600
#endif

#define THREADS 512

//smallest multiple of threadsPerBlock that is greater than or equal to N
#define BLOCKS min(32,(N*N+THREADS-1)/THREADS)



// Allocate the host input matrix A. Initialize Raw major
real_t h_A[N*N];

// Allocate the host input matrix B. Initialize Column Major
real_t h_B[N*N];

// Allocate the host output matrix C. Results Row Major
real_t h_C[N*N];

//can specify N to be x at compile-time as: gcc -DN=x

void host_matrixRandInitialize(int dim, real_t *matrix)
{
    int i,j,k;
    
    for(i = 0;i < dim; i++)
    {
        for(j = 0;j < dim; j++)
        {
            matrix[i*dim + j] = 0.0f;
            for(k = 0;k < dim; k++)
            {
                //matrix[i*dim + j] = 1 + rand()/(float)RAND_MAX;
                matrix[i*dim + j] = 1.0f;
            }
        }
    }
    return;
}

void host_matrixMultiply(int dim,int use_transpose, const real_t *A, const real_t *B, real_t *C){
	int i,j,k;
	if(use_transpose==1){
	   for(i = 0;i < dim; i++)
        {
            for(j = 0;j < dim; j++)
            {
                C[i*dim + j] = 0.0f;
                for(k = 0;k < dim; k++)
                {
                    C[i*dim + j] += A[i*dim + k] * B[j*dim + k];
                    //B is initialized column major, so j,k
                }
            }
        }
    }
    else{
       for(i = 0;i < dim; i++)
        {
            for(j = 0;j < dim; j++)
            {
                C[i*dim + j] = 0.0f;
                for(k = 0;k < dim; k++)
                {
                    C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
                    //B is initialized column major, so j,k
                }
            }
        }
    }
    return;
} 

float elapsed_time_msec(struct timespec *begin, struct timespec *end, unsigned long *sec, unsigned long *nsec){
    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec -1;
    }
    else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000;
}


/**
 * CUDA Kernel Device code
 *
 * Computes the matrix multiplication of A and B. The two matrices have the same
 * number of elements NxN.
 
 */

__global__ void matrixMultiply(int dim, const real_t *A, const real_t *B, real_t *C){

    //Each row is calculated by a block
    int i = blockIdx.x;
    //Each element in a row is calculated by a thread
    int j = threadIdx.x;
    
    while(i < dim){
        
        while(j < dim){
            C[i*dim + j] = 0.0f;
        
            for(int k = 0; k < dim; k++){
                C[i*dim + j] += A[i*dim + k] * B[j*dim + k];
            }
            //If thread count is less than N get another point        
            j += blockDim.x;
        }
        //If block count is less than N, get another row
        j = threadIdx.x;
        i += gridDim.x;
    }
}


 __global__ void tiledMatrixMultiply(int dim, const real_t *A,const real_t *B, real_t *C) {
         
         float CValue = 0;

         int row_index = blockIdx.y*blockDim.x + threadIdx.y;
         int column_index = blockIdx.x*blockDim.x + threadIdx.x;

         __shared__ real_t matA[THREADS*THREADS];
         __shared__ real_t matB[THREADS*THREADS];

         for (int k = 0; k < (blockDim.x + dim - 1)/blockDim.x; k++) {
             if (k*blockDim.x + threadIdx.x < dim && row_index < dim){

                     matA[threadIdx.y][threadIdx.x] = A[row_index*dim + k*blockDim.x + threadIdx.x];
             }
             else{

                     matA[threadIdx.y][threadIdx.x] = 0.0;
             }

             if (k*blockDim.x + threadIdx.y < dim && column_index < dim){
                     
                     matB[threadIdx.y][threadIdx.x] = B[(k*blockDim.x + threadIdx.y)*dim + column_index];
             }
             else{

                     matB[threadIdx.y][threadIdx.x] = 0.0;
             }

             // Wait till all the threads finish before calculating the results
             __syncthreads();

             for (int n = 0; n < blockDim.x; ++n){
                     CValue += matA[threadIdx.y][n] * matB[n][threadIdx.x];
             }
             __syncthreads();
         }

        if (row_index < dim && column_index < dim){
                 C[((blockIdx.y * blockDim.y + threadIdx.y)*dim) + (blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
        }
 }


int main(int argc, char **argv)
{
	int numthreads = 1;
	struct timespec t1_gpu, t2_gpu, t1_host, t2_host;
	unsigned long sec, nsec;
	float host_time,gpu_time; // in milli seconds


   int pflag = 0;
   int vflag = 0;
   int cflag = 0;
   int tflag = 0;
   int index;
   int c;

   opterr = 0;

   while ((c = getopt (argc, argv, "pvct")) != -1)
     switch (c)
       {
       case 'p':
         pflag = 1;
         break;
       case 'v':
         vflag = 1;
         break;
       case 'c':
         cflag = 1;
         break;
       case 't':
         tflag = 1;
         break;
       case '?':
         if (isprint(optopt))
           fprintf (stderr, "Unknown option `-%c'.\n", optopt);
         else
           fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
         return 1;
       default:
         abort ();
       }

   printf ("pflag = %d, vflag = %d, cflag = %d tflag = %d\n", pflag, vflag, cflag,tflag);

   for (index = optind; index < argc; index++){
     if(pflag!=0 && index == 2){
     	numthreads = atoi(argv[index]);
     }
     else{
     	printf ("Unknown Non-option argument %s at %d\n", argv[index],index);
     }
   }
   

	// do initializations, setting-up etc
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = N * N * sizeof(real_t);
    printf("[Matrix multiplication of %dx%d elements using %d blocks %d threads per block]\n", N,N,BLOCKS,THREADS);


    host_matrixRandInitialize(N,h_A);
    host_matrixRandInitialize(N,h_B);
    
    

    GET_TIME(t1_gpu);
	// do computation
	
	//cudaSetDevice(1);

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if(tflag){
        tiledMatrixMultiply<<<BLOCKS, THREADS>>>(N, (const real_t *)d_A,(const real_t *)d_B,(real_t *)d_C);
    }else{
	   matrixMultiply<<<BLOCKS, THREADS>>>(N, (const real_t *)d_A,(const real_t *)d_B,(real_t *)d_C);        
    }


	err = cudaMemcpy ( h_C , d_C , N * N * sizeof(real_t) , cudaMemcpyDeviceToHost ) ;

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	GET_TIME(t2_gpu);

    //printf("(0,0) %.2f \n",h_C[0]);
    for(int i = 0;i < 10; i++)
    {
        for(int j = 0;j < 10; j++)
        {
            printf("%.2f ",h_C[i*N + j]);
        }
        printf("\n");
    }

	GET_TIME(t1_host);
    if(tflag){
        //for tiling, initialized as regular
        host_matrixMultiply(N,0, (const real_t *)&h_A,(const real_t *)&h_B,(real_t *)&h_C);
	}
    else{
        //use transpose of B
        host_matrixMultiply(N,1, (const real_t *)&h_A,(const real_t *)&h_B,(real_t *)&h_C);
    }
    GET_TIME(t2_host);
	host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
	

    // printf("=====================");
    // for(int i = 0;i < N; i++)
    // {
    //     for(int j = 0;j < N; j++)
    //     {
    //         printf("%.2f ",h_C[i*N + j]);
    //     }
    //     printf("\n");
    // }



	gpu_time = elapsed_time_msec(&t1_gpu, &t2_gpu, &sec, &nsec);
	printf("N=%ld: Threads=%d: Time(ms)=%.2f \n", (long)N, numthreads, gpu_time);
	
	printf("CPU Serial Time(ms)=%.2f \n", host_time);
	printf("GPU Time(ms)=%.2f \n", gpu_time);
	
	// finishing stuff
	// Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");


}

