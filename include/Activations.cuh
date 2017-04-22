#ifndef ACTIVATIONS_CUH_
#define ACTIVATIONS_CUH_
#include <cuda.h>
#include <stdio.h>
#define BLOCK_XL 512
#define BLOCK_L 32
#define BLOCK_S 4

enum Activation{SIGMOID,TANH,RELU,ELU,LINEAR,LOGISTIC};

__global__ void launch_activations_on_gpu(float *x, int numElems, Activation a, float *y);
__global__ void launch_gradients_on_gpu(float *x, Activation a, int numElems, float *delta);

__device__ float select_activation(float x, Activation a);
__device__ void select_gradient(float x, Activation a);

void activations_on_gpu(float *x, int numelems, Activation a, float *y_data);
void gradient_on_gpu(float *x, int numElems, Activation a, float *delta);


#endif
