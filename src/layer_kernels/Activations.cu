#include <stdio.h>
#include <cuda.h>
#include "../include/Activations.cuh"

/* activation kernels */
__device__ float sigmoid_activation_kernel(float x){
  return 1/(1+exp(-x));
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


__device__ float linear_activation_kernel(float x){return x};
__device__ float logistic_activation_kernel(float x){return 1./(1.+exp(-x));}

/* gradient kernels */

__device__ float sigmoid_gradient_kernel(float x){

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


__device__ select_activation(float x, Activation a) {
  switch (a) {
    case SIGMOID:
           sigmoid_activation_kernel(x);
           break;
    case TANH:
           tanh_activation_kernel(x);
           break;
    case RELU:
          relu_activation_kernel(x);
          break;
    case ELU:
          elu_activation_kernel(x);
          break;
    case LINEAR:
          linear_activation_kernel(x);
          break;
    case LOGISTIC:
          logistic_activation_kernel(x);
          break;
    default:
          relu_activation_kernel(x);
          break;
  }
}


__global__ void launch_activations_on_gpu(float *x,int numElems,Activation a){
    int blockId = blockIdx.x + gridDim.x*blockIdx.y;
    int i = blockId + threadIdx.x;
    if (i<numElems) {
      select_activation(x[i],a);
    }


}

void activations_on_gpu(float *x ,int numElems, Activation a){
  cudaError_t err;
  dim3 block(32,1);
  dim3 grid(16,16);
  launch_activations_on_gpu<<< grid, block >>> (x,numElems,a);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    pritnf("Can't launch the activation kernels.\n");
  }
}
