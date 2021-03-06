#include <stdio.h>
#include <cuda.h>

#include "../include/activations.cuh"

/* activation kernels */
__device__ float sigmoid_activation_kernel(float x){
   float y = 1./(1.+expf(-x));
  //  printf("%.2f ",y);
   return y;
}


__device__ float tanh_activation_kernel(float x) {
    return (2./(1+exp(-2*x))-1);
}

__device__ float relu_activation_kernel(float x){
  return x>0? x:0;
}

__device__ float elu_activation_kernel(float x){
  if (x>=0) {
    return x;
  } else {
    return (0.01*exp(-x)+1);
  }
}


__device__ float linear_activation_kernel(float x){return x;}
__device__ float logistic_activation_kernel(float x){return 1./(1.+exp(-x));}

/* gradient kernels */

__device__ float sigmoid_gradient_kernel(float x){
   return (-x*exp(-x));
}

__device__ float tanh_gradient_kernel(float x){
  if (x>-1 && x< 1) {
    return 1;
  } else{
    return 0;
  }
}


__device__ float relu_gradient_kernel(float x) {
  if (x>0) {
    return 1;
  } else {
    return 0;
  }
}


__device__ float elu_gradient_kernel(float x) {
  if (x>=0) {
    return 1;
  } else {
    return (-0.01*exp(-x));
  }
}

__device__ float logistic_gradient_kernel(float x){return (1-x)*x;}


__device__ float select_activation(float x, Activation a) {
  switch (a) {
    case SIGMOID:
           return sigmoid_activation_kernel(x);
           break;
    case TANH:
           return tanh_activation_kernel(x);
           break;
    case RELU:
          return relu_activation_kernel(x);
          break;
    case ELU:
          return elu_activation_kernel(x);
          break;
    case LINEAR:
          return linear_activation_kernel(x);
          break;
    case LOGISTIC:
          return logistic_activation_kernel(x);
          break;
    default:
          return relu_activation_kernel(x);
          break;
  }
  return 0;
}


__global__ void launch_activations_on_gpu(float *x,int numElems,Activation a, float *y){
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if(i < numElems) {
      y[i] = select_activation(x[i],a);
      // printf("i:%d, %.2f \n",i,y[idx]);
      // id += gridDim.x*blockIdx.x;
    }


}

void activations_on_gpu(float *x, int numelems, Activation a, float *y_data){
  cudaError_t err;
  int k = (numelems-1)/BLOCK_L+1;
  int y = 1;
  int x_dir = k;

  dim3 grid(x_dir,y,1);
  // printf("Launcing the requested kernels...\n");
  launch_activations_on_gpu<<< grid, BLOCK_L>>> (x,numelems,a,y_data);
  err = cudaGetLastError();
  // printf("err: %d", err);
  if (err != cudaSuccess) {
    printf("Can't launch the activation kernels.\n");
  }
}
