// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/PiMyRandom.cu

// Written by Barry Wilkinson, UNC-Charlotte. PiMyRandom.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>


#define CSV_OUTPUT


int TRIALS_PER_THREAD = 4096;

#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi

#ifdef DP
	typedef double real_t;
#else
	typedef float real_t;
#endif
	
__device__ float my_rand(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
        unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

        return ((float)x)/m;
}

__global__ void gpu_monte_carlo(int trials_per_thread, real_t *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	real_t x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < trials_per_thread; i++) {
		x = my_rand(&seed);
		y = my_rand(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (real_t) trials_per_thread; // return estimate of pi
}

real_t host_monte_carlo(long trials) {
	real_t x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	real_t host[BLOCKS * THREADS];
	real_t *dev;

	if(argc > 1){
		TRIALS_PER_THREAD = atoi(argv[1]);
	}
	
	#ifdef CSV_OUTPUT
			printf("[MYRAND],precision,trials/thread,blocks,threads/block,gpu-pi-time,cpu-pi-time,gpu-pi,gpu-error,cpu-pi,cpu-error,\n");
			printf("[MYRAND],");

		#ifdef DP
			printf("dp,");
		#else
			printf("sp,");
		#endif
			printf("%d,",TRIALS_PER_THREAD);
			printf("%d,",BLOCKS);
			printf("%d,",THREADS);
	#else
	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);
	#endif
	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(real_t)); // allocate device mem. for counts

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(TRIALS_PER_THREAD,dev);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(real_t), cudaMemcpyDeviceToHost); // return results 

	real_t pi_gpu;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

	#ifdef CSV_OUTPUT
		printf("%f,",(stop-start)/(float)CLOCKS_PER_SEC);
	#else
		printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	#endif
	start = clock();
	real_t pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	#ifdef CSV_OUTPUT
		printf("%f,",(stop-start)/(float)CLOCKS_PER_SEC);
	#else
		printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	#endif

	#ifdef CSV_OUTPUT
			printf("%f,",pi_gpu);
			printf("%f,",pi_gpu - PI);
			printf("%f,",pi_cpu);
			printf("%f,\n",pi_cpu - PI);
			
	#else

		printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	#endif

	return 0;
}
