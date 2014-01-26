#include <stdio.h>
#include <time.h> 	// for clock_gettime( )
#include <errno.h>	// for perror( )
#include <getopt.h>	// for getopt( )
#include <ctype.h>	//for isprint( )
#include <pthread.h>

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
#define TILE_DIM 32

//smallest multiple of threadsPerBlock that is greater than or equal to N
#define BLOCKS min(32,(N*N+THREADS-1)/THREADS)

typedef struct{
    int thread_id;
    int threadcount;
    long dim;
    const real_t *A;
    const real_t *B;
    real_t *C;
}try_arg_t;


// Allocate the host input matrix A. Initialize Raw major
real_t h_A[N*N];

// Allocate the host input matrix B. Initialize Column Major
real_t h_B[N*N];

// Allocate the host output matrix C. Results Row Major
real_t h_C[N*N];

// Allocate the output matrix Cd copied from device. Results Row Major
real_t h_Cd[N*N];

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

void * host_matrixMultiply_per_Thread(void * arg){
    
    try_arg_t *args = (try_arg_t *)arg;
    int thread_id = args->thread_id;
    int threadcount = args->threadcount;    
    long dim = args->dim;
    const real_t *A = args->A;
    const real_t *B = args->B;
    real_t *C = args->C;

    //Each row is calculated by a thread
    long c_index = thread_id;
   
    printf("Created thread: %d, dim:%ld\n",thread_id,dim);

    while(c_index < dim*dim){
        
        printf("c_index: %ld\n",c_index);
        
        long i=c_index/dim;
        long j = c_index % dim;
        printf("c_index: %ld, i:%ld, j:%ld, Aval:%lf Bval:%lf\n",c_index,i,j,A[i*dim + 0],B[j]);

        //C[c_index] = 0.0f;
        printf("c_index: %ld, i:%ld, j:%ld, Aval:%lf Bval:%lf\n",c_index,i,j,A[i*dim + 0],B[j]);

        for(long k = 0; k < dim; k++){
            printf("calculating index:%ld, i:%ld, j:%ld, k:%ld\n",c_index,i,j,k);
            C[c_index] += A[i*dim + k] * B[k*dim + j];
        }
        
        //If thread count is less than dim*dim, get another row
        i += threadcount;
    }
    return 0;
}

void host_pthread_MatrixMultiply(int num_pthreads, long dim, const real_t *A, const real_t *B, real_t *C){
    long t;
    int rc;
    pthread_t *threads;
    pthread_attr_t attr;
    try_arg_t *try_args;
    void * status;
    
    try_args = (try_arg_t *)malloc(num_pthreads*sizeof(try_arg_t));
    if(try_args == NULL){
        printf("ERROR; return malloc failed for try_args");
    }
    threads = (pthread_t *)malloc(num_pthreads*sizeof(pthread_t));  //  Allocate pthreads
    if(threads == NULL){
        printf("ERROR; return malloc failed for threads");
    }

    C[0] = 0.0f;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    
    for(t = 0; t < num_pthreads; t++){
        try_args[t].thread_id = t;
        try_args[t].threadcount = num_pthreads;
        try_args[t].dim = dim;
        try_args[t].A = A;
        try_args[t].B = B;
        try_args[t].B = C;
        
        printf("Creating thread:%ld\n",t);
        rc = pthread_create(&threads[t],&attr,host_matrixMultiply_per_Thread,(void *)&try_args[t]);
        if(rc){
            printf("ERROR; return code from pthread_create()\
                 is %d\n", rc);
            exit(-1);
        }
    }
    pthread_attr_destroy(&attr);

    for(t = 0; t < num_pthreads; t++){
        //printf("pthread_join: ThreadID:%d\n",t);
        rc = pthread_join(threads[t], &status);
        //printf("pthread_joined: ThreadID:%d\n",t);
    }

    free(try_args);
    free(threads);
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

//Kernel taken from http://www.orangeowlsolutions.com/archives/526
__global__ void MatMul(real_t* A, real_t* B, real_t* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

  real_t CValue = 0; 
  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ real_t As[TILE_DIM][TILE_DIM];
  __shared__ real_t Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

      if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows) As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
      else As[threadIdx.y][threadIdx.x] = 0.0;

      if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)  Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
      else Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();

  }

  if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

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

    if(!tflag){
        printf("[Matrix multiplication of %dx%d elements using %d blocks %d threads per block]\n", N,N,BLOCKS,THREADS);
    }

    host_matrixRandInitialize(N,h_A);
    host_matrixRandInitialize(N,h_B);
    
    

    GET_TIME(t1_gpu);
	// do computation
	
	//cudaSetDevice(1);

    // Allocate the device input vector A
    real_t *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    real_t *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    real_t *d_C = NULL;
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


        // tile dimensions. Limitation: C [DIMX][DIMZ] = A [DIMX][DIMY] * B [DIMY][DIMZ] 

        int CCols = N, CRows=N, ACols=N, ARows=N, BCols=N, BRows=N;

        dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
        dim3 dimGrid;
     
        dimGrid.x = (CCols + dimBlock.x - 1)/dimBlock.x;
        dimGrid.y = (CRows + dimBlock.y - 1)/dimBlock.y;
 
        printf("[Matrix multiplication of %dx%d elements using %dx%d blocks %dx%d threads per block]\n", N,N,dimBlock.x,dimBlock.y,TILE_DIM,TILE_DIM);

        MatMul<<<dimGrid , dimBlock>>>(d_A , d_B , d_C , ARows , ACols, BRows ,BCols , CRows , CCols);
  
    }else{
	   matrixMultiply<<<BLOCKS, THREADS>>>(N, (const real_t *)d_A,(const real_t *)d_B,(real_t *)d_C);        
    }


	err = cudaMemcpy ( h_Cd , d_C , N * N * sizeof(real_t) , cudaMemcpyDeviceToHost ) ;

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	GET_TIME(t2_gpu);

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
    //released GPU




    //printf("(0,0) %.2f \n",h_C[0]);
    for(int i = 0;i < 10; i++)
    {
        for(int j = 0;j < 10; j++)
        {
            printf("%.2f ",h_Cd[i*N + j]);
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
	

    if(vflag==1){
    for (int i = 0; i<N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            real_t error = h_Cd[i*N+j] - h_C[i*N+j];
            if(error > 0.1 || error < -0.1)
            {
                printf("Matrix-matrix multiplication: unsuccessful... :-( \n");
            }
        }
    }
        printf("Verified results on +/-0.1 error margin\n");
    }
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
	

    
    GET_TIME(t1_host);
    host_pthread_MatrixMultiply(2,N,(const real_t *)&h_A,(const real_t *)&h_B,(real_t *)&h_C);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread (2threads) Time(ms)=%.2f \n", host_time);
    
    GET_TIME(t1_host);
    host_pthread_MatrixMultiply(4,N,(const real_t *)&h_A,(const real_t *)&h_B,(real_t *)&h_C);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread (4threads) Time(ms)=%.2f \n", host_time);
    
    GET_TIME(t1_host);
    host_pthread_MatrixMultiply(8,N,(const real_t *)&h_A,(const real_t *)&h_B,(real_t *)&h_C);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread (8threads) Time(ms)=%.2f \n", host_time);
    


	
    printf("Done\n");


}

