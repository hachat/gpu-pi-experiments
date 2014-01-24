// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>
#include <prng.h>

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
	
__global__ void gpu_monte_carlo(int tries_per_thread, real_t *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	real_t x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < tries_per_thread; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (real_t) tries_per_thread; // return estimate of pi
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

typedef enum{
	PRNG,
	RAND,
}random_generator_t;

typedef struct{
	int thread_id;
	long int ncount;
	random_generator_t rng_type;
	real_t estimate;
}try_arg_t;



void * parallel_monte_carlo_try(void * arg){
	//struct prng *g;
	try_arg_t *args = (try_arg_t *)arg;
	long trials = args->ncount;
	
	// if(args->rng_type == PRNG){
	// 	g = prng_new("eicg(2147483647,111,1,0)");
	// 	if(g == NULL){
	// 		printf("Initializing random number generator failed\n");
	// 		pthread_exit(NULL);
	// 	}
	// }
	//printf("Thread %d doing %d tries\n",t,ncount);

	real_t x, y;
	long points_in_circle = 0;

	//printf("trying: ThreadID:%d, try_count:%ld\n",args->thread_id,trials);
		
	for(long i = 0; i < trials; i++) {
		if(args->rng_type == RAND){
			x = rand() / (float) RAND_MAX;
			y = rand() / (float) RAND_MAX;
		}
		else if(args->rng_type == PRNG){
			x = rand() / (float) RAND_MAX;
			y = rand() / (float) RAND_MAX;
			// x = prng_get_next(g);
			// y = prng_get_next(g);
		}
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	args->estimate = 4.0f * points_in_circle / trials;

	// if(args->rng_type == PRNG){
	// 	prng_reset(g);
	// 	prng_free(g);
	// }

	//printf("finished: ThreadID:%d, try_count:%ld, estimate:%f\n",args->thread_id,trials,args->estimate);
	
	pthread_exit(NULL);	 
}


real_t host_pthread_monte_carlo(long trials,int num_pthreads,random_generator_t rng_type){
	long tries_per_pthread = 0;
	long t;
	int rc;
	pthread_t *threads;
	pthread_attr_t attr;
	try_arg_t *try_args;
	void * status;
	real_t pi_pthreads;

	try_args = (try_arg_t *)malloc(num_pthreads*sizeof(try_arg_t));
	if(try_args == NULL){
		printf("ERROR; return malloc failed for try_args");
	}
	threads = (pthread_t *)malloc(num_pthreads*sizeof(pthread_t));  //  Allocate pthreads
	if(threads == NULL){
		printf("ERROR; return malloc failed for threads");
	}
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
	tries_per_pthread = trials / num_pthreads;

	for(t = 0; t < num_pthreads; t++){
		try_args[t].thread_id = t;
		try_args[t].ncount = tries_per_pthread;
		try_args[t].rng_type = rng_type;
		try_args[t].estimate = 0.0f;//For output
		if(t == 0){
			//First thread. fixing integer division mismatch by adding the rest to
			//this thread. not actually much of an issue, as this is a random experiment
			try_args[t].ncount = trials - (tries_per_pthread*(num_pthreads-1));
		}
		
		//printf("pthread_create: ThreadID:%d, try_count:%ld\n",try_args[t].thread_id,try_args[t].ncount);
		rc = pthread_create(&threads[t],&attr,parallel_monte_carlo_try,(void *)&try_args[t]);
		if(rc){
			printf("ERROR; return code from pthread_create()\
				 is %d\n", rc);
			exit(-1);
		}

	}
	pthread_attr_destroy(&attr);

	pi_pthreads = 0.0f;
	for(t = 0; t < num_pthreads; t++){
		//printf("pthread_join: ThreadID:%d\n",t);
		rc = pthread_join(threads[t], &status);
		//printf("pthread_joined: ThreadID:%d\n",t);
		
		pi_pthreads += try_args[t].estimate;
	}
	pi_pthreads = pi_pthreads/num_pthreads;

	free(try_args);
	free(threads);
	return pi_pthreads;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	real_t host[BLOCKS * THREADS];
	real_t *dev;
	curandState *devStates;

	if(argc >1){
		TRIALS_PER_THREAD = atoi(argv[1]);
	}
	
	#ifdef CSV_OUTPUT
			if(argc >2){
				printf("[CURAND],precision,trials/thread,blocks,threads/block,gpu-pi-time,cpu-pi-time,pthread-pi-time,gpu-pi,gpu-error,cpu-pi,cpu-error,pthread-pi,pthread-error,pthread-count,\n");
			}
			else{
				printf("[CURAND],precision,trials/thread,blocks,threads/block,gpu-pi-time,cpu-pi-time,gpu-pi,gpu-error,cpu-pi,cpu-error,\n");	
			}
			printf("[CURAND],");

		#ifdef DP
			printf("dp,");
		#else
			printf("sp,");
		#endif
			printf("%d,",TRIALS_PER_THREAD);
			printf("%d,",BLOCKS);
			printf("%d,",THREADS);
	#else
	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD, BLOCKS, THREADS);
	#endif


	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(real_t)); // allocate device mem. for counts
	
	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(TRIALS_PER_THREAD, dev, devStates);

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

	real_t pi_cpu_pthread;
	int num_pthreads = 0;
	random_generator_t rng_type = RAND;
	if(argc >2){
		num_pthreads = atoi(argv[2]);
		if(argc >3){
			if(strcmp(argv[3],"RAND")){
				rng_type = RAND;
			}else if(strcmp(argv[3],"PRNG")){
				printf("PRNG Not Supported at the moment. reverting to RAND\n");
				rng_type = PRNG;
			}else{
				rng_type = RAND;
			}
		}
		start = clock();
		pi_cpu_pthread = host_pthread_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD,num_pthreads,rng_type);
		stop = clock();

		#ifdef CSV_OUTPUT
			printf("%f,",(stop-start)/(float)CLOCKS_PER_SEC);
			
		#else
			printf("CPU Pthread pi calculated in %f s. Used %d threads.\n", (stop-start)/(float)CLOCKS_PER_SEC,num_pthreads);
		#endif
	}

	#ifdef CSV_OUTPUT
			printf("%f,",pi_gpu);
			printf("%f,",pi_gpu - PI);
			printf("%f,",pi_cpu);
			printf("%f,",pi_cpu - PI);
			
	#else

		printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	#endif
	if(argc >2){
		#ifdef CSV_OUTPUT
			printf("%f,",pi_cpu_pthread);
			printf("%f,",pi_cpu_pthread - PI);
			printf("%d,\n",num_pthreads);
		#else
			printf("CPU pthread estimate of PI = %f [error of %f]\n", pi_cpu_pthread, pi_cpu_pthread - PI);
		#endif
	}
	return 0;
}

