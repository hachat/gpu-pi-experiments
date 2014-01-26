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
#define N 10000000
#endif

#define THREADS 512

//smallest multiple of threadsPerBlock that is greater than or equal to N
#define BLOCKS min(32,(N+THREADS-1)/THREADS)

typedef struct{
    int thread_id;
    int threadcount;
    long totalLength;
    real_t *threadSum;
    const real_t *A;
    const real_t *B;
}try_arg_t;


    // Allocate the host input vector A
    real_t h_A[N];

    // Allocate the host input vector B
    real_t h_B[N];

    // Allocate the host output vector C
    real_t h_C[BLOCKS];

//can specify N to be x at compile-time as: gcc -DN=x

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
 * Computes the vector multiplication of A and B. The two vectors have the same
 * number of elements numElements.
 */
 //http://cuda-programming.blogspot.in/2013/01/vector-dot-product-in-cuda-c-cuda-c.html
__global__ void
vectorMultiply(const real_t *A, const real_t *B, real_t *C)
{

    unsigned long data_index = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long block_sum_index = threadIdx.x;
	// This keeps the sum from each thread within the block
	__shared__ real_t blockSums[THREADS];

	real_t threadSum = 0;

	while ( data_index < N )
	{
		threadSum += A[data_index] * B[data_index];
		data_index += blockDim.x * gridDim.x;
	}
	blockSums[threadIdx.x] = threadSum;
	//printf("BlockID: %d ThreadID: %d threadSum: %lf\n",blockIdx.x , threadIdx.x,threadSum);
	
	__syncthreads();

	long i = blockDim.x/2;
	//Two by two iterative reduction 
	while (i!=0){
		if(block_sum_index < i){
			//Lower half
			blockSums[block_sum_index] += blockSums[block_sum_index + i] ;
		}
		__syncthreads();

		i = i/2;
	}

	if(block_sum_index == 0){
		C[blockIdx.x] = blockSums[0];
	}
}

real_t host_vectorMultiply(const real_t *A, const real_t *B){
	long i = 0;
	real_t dotProduct = 0.0f;

	for(i = 0;i < N; i++)
    {
        dotProduct += A[i] * B[i];
    }
    return dotProduct;
} 

void * host_vectorMultiply_per_Thread(void * arg){
    
    try_arg_t *args = (try_arg_t *)arg;
    int thread_id = args->thread_id;
    int threadcount = args->threadcount;
    long totalLength = args->totalLength;
    real_t *threadSum = args->threadSum;
    const real_t *A = args->A;
    const real_t *B = args->B;
    long i = 0;
    real_t dotProduct = 0.0f;

    long begin = thread_id * (totalLength/threadcount);
    long end  = (thread_id + 1) * (totalLength/threadcount);
    if (end > N)
    {
        end = N;
    }
    for(i = begin;i < end; i++)
    {
        dotProduct += A[i] * B[i];
    }
    *threadSum = dotProduct;
    //printf("thread :%d, from %ld to %ld : %f\n",thread_id,begin,end,dotProduct);
    
    return 0;
} 

real_t host_pthread_vectorMultiply(int num_pthreads,const real_t *A, const real_t *B){
    long t;
    int rc;
    pthread_t *threads;
    pthread_attr_t attr;
    try_arg_t *try_args;
    void * status;
    real_t *threadSums;


    real_t dotProduct = 0.0f;

    try_args = (try_arg_t *)malloc(num_pthreads*sizeof(try_arg_t));
    if(try_args == NULL){
        printf("ERROR; return malloc failed for try_args");
    }
    threads = (pthread_t *)malloc(num_pthreads*sizeof(pthread_t));  //  Allocate pthreads
    if(threads == NULL){
        printf("ERROR; return malloc failed for threads");
    }
    threadSums = (real_t *)malloc(num_pthreads*sizeof(real_t));  //  Allocate pthreads
    if(threadSums == NULL){
        printf("ERROR; return malloc failed for threadSums");
    }

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    
    for(t = 0; t < num_pthreads; t++){
        try_args[t].thread_id = t;
        try_args[t].threadcount = num_pthreads;
        try_args[t].totalLength = N;
        try_args[t].threadSum = &threadSums[t];
        try_args[t].A = h_A;
        try_args[t].B = h_B;
        rc = pthread_create(&threads[t],&attr,host_vectorMultiply_per_Thread,(void *)&try_args[t]);
        if(rc){
            printf("ERROR; return code from pthread_create()\
                 is %d\n", rc);
            exit(-1);
        }

    }
    pthread_attr_destroy(&attr);


    dotProduct = 0.0f;
    for(t = 0; t < num_pthreads; t++){
        //printf("pthread_join: ThreadID:%d\n",t);
        rc = pthread_join(threads[t], &status);
        //printf("pthread_joined: ThreadID:%d\n",t);
        
        dotProduct += threadSums[t];
    }

    free(try_args);
    free(threads);
    free(threadSums);
    return dotProduct;
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
   int index;
   int c;

   opterr = 0;

   while ((c = getopt (argc, argv, "pvc")) != -1)
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

   printf ("pflag = %d, vflag = %d, cflag = %d\n", pflag, vflag, cflag);

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


    size_t size = N * sizeof(real_t);
    printf("[Vector multiplication of %ld elements using %d blocks %d threads per block]\n", (long)N,BLOCKS,THREADS);

    // Initialize the host input vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1 + rand()/(float)RAND_MAX;
        h_B[i] = 1 + rand()/(float)RAND_MAX;
    }
    

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
    err = cudaMalloc((void **)&d_C, BLOCKS * sizeof(real_t));

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

	vectorMultiply<<<BLOCKS, THREADS>>>((const real_t *)d_A,(const real_t *)d_B,(real_t *)d_C);


	err = cudaMemcpy ( h_C , d_C , BLOCKS * sizeof(real_t) , cudaMemcpyDeviceToHost ) ;

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    real_t gpu_dotProduct = 0.0f;

	for (int i = 0; i < BLOCKS; i++)
    {
        gpu_dotProduct += h_C[i];
        //printf("BlockID:%d : BlockSum:%lf\n",i,h_C[i]);
    }


	GET_TIME(t2_gpu);

	GET_TIME(t1_host);
    real_t host_dotProduct = host_vectorMultiply((const real_t *)&h_A,(const real_t *)&h_B);
	GET_TIME(t2_host);
	host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
	
	gpu_time = elapsed_time_msec(&t1_gpu, &t2_gpu, &sec, &nsec);
	printf("N=%ld: Threads=%d: Time(ms)=%.2f \n", (long)N, numthreads, gpu_time);
	

	printf("CPU Serial dotProduct: %lf\n",host_dotProduct);
	printf("CPU Serial Time(ms)=%.2f \n", host_time);
	
    printf("GPU dotProduct: %lf\n",gpu_dotProduct);
    printf("GPU Time(ms)=%.2f \n", gpu_time);
    
    real_t host_pthread_dotProduct;
    
    GET_TIME(t1_host);
    host_pthread_dotProduct = host_pthread_vectorMultiply(2,(const real_t *)&h_A,(const real_t *)&h_B);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread dotProduct(2threads): %lf\n",host_pthread_dotProduct);
    printf("CPU pthread Time(ms)=%.2f \n", host_time);
    
    GET_TIME(t1_host);
    host_pthread_dotProduct = host_pthread_vectorMultiply(4,(const real_t *)&h_A,(const real_t *)&h_B);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread dotProduct(4threads): %lf\n",host_pthread_dotProduct);
    printf("CPU pthread Time(ms)=%.2f \n", host_time);
    
    GET_TIME(t1_host);
    host_pthread_dotProduct = host_pthread_vectorMultiply(8,(const real_t *)&h_A,(const real_t *)&h_B);
    GET_TIME(t2_host);
    host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
    
    printf("CPU pthread dotProduct(8threads): %lf\n",host_pthread_dotProduct);
    printf("CPU pthread Time(ms)=%.2f \n", host_time);
    
	// finishing stuff
	// Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
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

