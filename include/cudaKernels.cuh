#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_
#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>

__global__ void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols);
__global__ void WeightedAddKernel(float *a, float*b, float*c, float scale, int numRows, int numCols);
__global__ void EltWiseMatMul(float *a, float *b, float *c, int numRows, int numCols);
__global__ void EltWiseMatDivide(float *a, float *b, float *c, int numRows, int numCols);
__global__ void powgpu_kernel(float *a, int n, int scale);
__global__ void expgpu_kernel(float *a, int n);
__global__ void axpy_kernel(float *a, float *b, float scaleA, int lda, int ldy, int n,float *y);
#endif
