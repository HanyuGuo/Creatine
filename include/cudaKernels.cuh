#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_
#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>

__global__ void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols);
__global__ void WeightedAddKernel(float *a, float*b, float*c, float scale, int numRows, int numCols);
__global__ void EltWiseMatMul(float *a, float *b, float *c, int numRows, int numCols);
__global__ void EltWiseMatDivide(float *a, float *b, float *c, int numRows, int numCols);
#endif
