#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_


#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define max(a,b) (a>b?a:b)

__global__ void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols);
__global__ void WeightedAddKernel(float *a, float*b, float*c, float scale, int numRows, int numCols);
__global__ void EltWiseMatMul(float *a, float *b, float *c, int numRows, int numCols);
__global__ void EltWiseMatDivide(float *a, float *b, float *c, int numRows, int numCols);
__global__ void powgpu_kernel(float *a, int n, int scale);
__global__ void expgpu_kernel(float *a, int n, float *y);
__global__ void axpy_kernel(float *a, float *b, float scaleA, int lda, int ldy, int n,float *y);
__global__ void add_mat_vec_kernel(float *a, float *b, int nr, int nc, float scale, float *y);
__global__ void DivideByScalar(float *x,float *scalar, int nr, int nc);
__global__ void ReduceSum(float *x, float *y,int nr, int nc);
__global__ void calc_max(int bs, int nc, float *x, float *y); // calc the max;
__global__ void subtract_max(int bs,int nc, float *x, float *max, float *y); // subtract the max;
__global__ void calc_sum_row(int bs, int nc, float *x, float *r_sum); //calculate the sum of row
__global__ void div_row(int bs, int nc, float *x, float *row, float *y); //calc the division per row.

// __device__ void softmax_device(float *x, int n, int stride, float *y); // inspired by pjreddie/Darknet.
// __global__ void softmax_kernel(float *x, int n,int nc, int stride, float *y);



#endif
