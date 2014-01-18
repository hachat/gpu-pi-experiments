// Source: http://docs.nvidia.com/cuda/curand/index.html

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>

#include <time.h>

// we could vary M & N to find the perf sweet spot
int TRIALS_PER_THREAD = 4096;

#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi

#ifdef DP
  typedef double real_t;
#else
  typedef float real_t;
#endif
  
struct estimate_pi : 
    public thrust::unary_function<unsigned int, real_t>
{
  __device__
  real_t operator()(unsigned int thread_id)
  {
    real_t sum = 0;
    unsigned int N = TRIALS_PER_THREAD; // samples per thread

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      real_t x = curand_uniform(&s);
      real_t y = curand_uniform(&s);

      // measure distance from the origin
      real_t dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main(int argc, char *argv[])
{
  // use 30K independent seeds
  int M = BLOCKS*THREADS;
  clock_t start, stop;
  
  if(argc > 1){
    TRIALS_PER_THREAD = atoi(argv[1])
  }
  std::cout << "# of trials per thread = "<< TRIALS_PER_THREAD <<" # of blocks * # of threads/block = " 
            << BLOCKS*THREADS << std::endl;

  start = clock();

  real_t estimate = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M),
        estimate_pi(),
        0.0f,
        thrust::plus<real_t>());
  estimate /= M;

  stop = clock();
  float elapsed_time = (stop-start)/(float)CLOCKS_PER_SEC;
  float error = estimate - PI;

  std::cout << std::setprecision(7);
  std::cout << "THRUST pi calculated in " << elapsed_time << " s."<< std::endl;

  std::cout << "CUDA estimate of PI = " << estimate << " [error of " << error << "]" << std::endl;
  
  return 0;
}

