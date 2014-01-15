// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <time.h>

#ifndef TRIALS_PER_THREAD
	#define TRIALS_PER_THREAD 4096
#endif
#define NBIN TRIALS_PER_THREAD  // Number of bins
#define NUM_BLOCK  256  // Number of thread blocks
#define NUM_THREAD  256  // Number of threads per block
#define PI 3.1415926535  // known value of pi

#ifdef DP
	typedef double real_t;
#else
	typedef float real_t;
#endif

int tid;
real_t pi_gpu = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(real_t *sum, int nbin, real_t step, int nthreads, int nblocks) {
	int i;
	real_t x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
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

// Main routine that executes on the host
int main(void) {
	clock_t start, stop;
	
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	real_t *sumHost, *sumDev;  // Pointer to host & device arrays

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", NBIN,
NUM_BLOCK, NUM_THREAD);

	start = clock();

	real_t step = 1.0/NBIN;  // Step size
	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(real_t);  //Array memory size
	sumHost = (real_t *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **) &sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	// Do calculation on device
	cal_pi <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
		pi_gpu += sumHost[tid];
	pi_gpu *= step;

	stop = clock();
	// Print results
	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// Cleanup
	free(sumHost); 
	cudaFree(sumDev);

	start = clock();
	real_t pi_cpu = host_monte_carlo(NUM_BLOCK * NUM_THREAD * NBIN);
	stop = clock();
	printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	
	return 0;
}
