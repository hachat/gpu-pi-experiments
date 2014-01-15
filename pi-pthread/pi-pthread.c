#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h> /* gettimeofday() */
#include <prng.h>

#define NUMTHREADS_MAX 20
#define TOTAL_TRIES 2000000
long int success_count[NUMTHREADS_MAX];
int numthreads = 1;
long int total_tries = 0;
typedef struct{
	int thread_id;
	long int ncount;
}try_arg;

int rnd_generator = 0;

try_arg try_args[NUMTHREADS_MAX];

//time_t clock_start,clock_end;
struct timeval start,end,diff;

/* Subtract the `struct timeval' values X and Y,
 *    storing the result in RESULT.
 *       Return 1 if the difference is negative, otherwise 0.  */

int
timeval_subtract (result, x, y)
     struct timeval *result, *x, *y;
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
 *      tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

void * try(void * arg){
	struct prng *g;
	try_arg *args = (try_arg *)arg;
	long int ncount = args->ncount;
	int t = args->thread_id;
	int i = 0;
	g = prng_new("eicg(2147483647,111,1,0)");
	if(g == NULL){
		printf("Initializing random number generator failed\n");
		pthread_exit(NULL);
	}
	//printf("Thread %d doing %d tries\n",t,ncount);
	for(i=0; i < ncount; i++){
		double xc,yc;
		if(rnd_generator == 0){
			xc = rand()*1.0/RAND_MAX;
			yc = rand()*1.0/RAND_MAX;
		}
		else if(rnd_generator == 1){
			xc = prng_get_next(g);
			yc = prng_get_next(g);
		}
		if((xc*xc) + (yc*yc) < 1.0){
			success_count[t] += 1;
		}
	}
	prng_reset(g);
	prng_free(g);
	pthread_exit(NULL);	 
}

int main(int argc, char * argv[]){
	pthread_t threads[NUMTHREADS_MAX];
	pthread_attr_t attr;


	double PI;
	long t;
	int rc;
	long int tries_per_thread = 0;
	long int total_tries = 0;
	long int total_successes = 0;
	void * status;
	if(argc < 2) {
		printf(" Usage: pi_pthreads <number of threads> <randoom generator 0-rand(), 1-prng eicg>\n");
		return 0;
	}

	numthreads = atoi(argv[1]);
	if(numthreads >NUMTHREADS_MAX){
		printf("Only upto %d Threads allowed\n",NUMTHREADS_MAX);	
	}
	if(argc > 2){
		rnd_generator = atoi(argv[2]);	
	}
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
	//clock_start = clock();
	tries_per_thread = TOTAL_TRIES/numthreads;
	gettimeofday(&start,NULL);
	for(t = 0; t < numthreads; t++){
		try_args[t].thread_id = t;
		try_args[t].ncount = tries_per_thread;
		rc = pthread_create(&threads[t],&attr,try,(void *)&try_args[t]);
		if(rc){
			printf("ERROR; return code from pthread_create()\
				 is %d\n", rc);
			exit(-1);
		}
	}
	pthread_attr_destroy(&attr);
	for(t = 0; t < numthreads; t++){
		rc = pthread_join(threads[t], &status);
		//printf("Joined thread %d\n",t);
		total_tries = total_tries + try_args[t].ncount;
		total_successes = total_successes + success_count[t]; 	
	}
	PI = total_successes * 4.0 / total_tries; 
	
	
	

	gettimeofday(&end,NULL);
	long long elapsed = (end.tv_sec-end.tv_sec)*1000000LL + end.tv_usec-start.tv_usec;
	//printf("TimeOfDay: %d:%d\n",start.tv_sec,start.tv_usec);
	//printf("Time of Day End: %d:%d\n",end.tv_sec,end.tv_usec);
	timeval_subtract(&diff,&end,&start);
   	//printf("Elapsed Time: %d:%d\n",diff.tv_sec,diff.tv_usec);
 	//printf("Number of threads: %d, Tries Per Thread : %d, Total Tries : %d,\
Total successes %d, PI Approximation: %lf, Elapsed Time: %d:%d\n" \
 	,numthreads,tries_per_thread,total_tries, total_successes,PI,diff.tv_sec,diff.tv_usec);      
	printf("%d,%d,%d,%d,%lf,%d:%d\n"\
	,numthreads,tries_per_thread,total_tries, total_successes,PI,diff.tv_sec,diff.tv_usec); 	
	pthread_exit(NULL);
	return 0;
}
