#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_
#include<stdio.h>
#include<cuda_runtime.h>


__global__ void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols);
void LaunchAddKernel(float *a, float *b, float *c, int numRows, int numCols);

#endif
