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
	
#define BLOCKS 256
#define THREADS 256

#ifndef N
#define N 10000000
#endif



    // Allocate the host input vector A
    real_t h_A[N];

    // Allocate the host input vector B
    real_t h_B[N];

    // Allocate the host output vector C
    real_t h_C[BLOCKS * THREADS];


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
__global__ void
vectorMultiply(const real_t *A, const real_t *B, real_t *C, long int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] += A[i] * B[i];
    }
}

real_t host_vectorMultiply(const real_t *A, const real_t *B, long int numElements){
	long int i = 0;
	real_t dotProduct = 0.0f;

	if (i < numElements)
    {
        dotProduct += A[i] * B[i];
    }
    return dotProduct;
} 

int main(int argc, char **argv)
{
	int numthreads = 1;
	struct timespec t0, t1, t2,t1_host,t2_host;
	unsigned long sec, nsec;
	float comp_time,host_time; // in milli seconds


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
   

	GET_TIME(t0);
	// do initializations, setting-up etc
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = N * sizeof(real_t);
    printf("[Vector multiplication of %ld elements using %d blocks %d threads per block]\n", (long)N,BLOCKS,THREADS);

    // Initialize the host input vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1 + rand()/(float)RAND_MAX;
        h_B[i] = 1 + rand()/(float)RAND_MAX;
    }
    // Initialize the host input vectors
    for (int i = 0; i < BLOCKS * THREADS; ++i)
    {
        h_C[i] = 0.0f;
    }


	GET_TIME(t1);
	// do computation
	GET_TIME(t2);

	GET_TIME(t1_host);
    real_t host_dotProduct = host_vectorMultiply((const real_t *)&h_A,(const real_t *)&h_B,N);
	GET_TIME(t2_host);
	host_time = elapsed_time_msec(&t1_host, &t2_host, &sec, &nsec);
	
	comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
	printf("N=%ld: Threads=%d: Time(ms)=%.2f \n", (long)N, numthreads, comp_time);
	
	printf("CPU Serial dotProduct: %lf\n",host_dotProduct);
	printf("CPU Serial Time(ms)=%.2f \n", host_time);
	
	// finishing stuff

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");


}

