#include<stdio.h>
#include<cuda.h>
#include "../include/cudaKernels.cuh"


__global__ void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols){
int idx = blockIdx.x*blockDim.x+threadIdx.x;
int idy = blockIdx.y*blockDim.y+threadIdx.y;
if(idy < numRows && idx < numCols) {
  c[idy*numRows+idx] = a[idy*numRows+idx]+b[idy*numRows+idx];
}
}

__global__ void WeightedAddKernel(float *a, float *b, float *c,float scale,int numRows,int numCols){
int idx = blockIdx.x*blockDim.x+threadIdx.x;
int idy = blockIdx.y*blockDim.y+threadIdx.y;
if(idy < numRows && idx < numCols) {
  c[idy*numRows+idx] = a[idy*numRows+idx]+ scale*b[idy*numRows+idx];
}
}

__global__ void EltWiseMatMul(float *a, float *b, float *c,int numRows, int numCols) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  if(idy < numRows && idx < numCols) {
    c[idy*numRows+idx] = a[idy*numRows+idx]*b[idy*numRows+idx];
  }
}


__global__ void EltWiseMatDivide(float *a, float *b, float *c,int numRows, int numCols) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  if(idy < numRows && idx < numCols) {
    c[idy*numRows+idx] = a[idy*numRows+idx]/b[idy*numRows+idx];
  }
}


__global__ void  powgpu_kernel(float *a, int n,  int scale){
  int id = (blockIdx.x + gridDim.x*blockIdx.y)*blockDim.x + threadIdx.x;
  if (id < n) {
    a[id] = pow(a[id],scale);
  }
}

__global__ void expgpu_kernel(float *a, int n){
  int id = (blockIdx.x + gridDim.x*blockIdx.y)*blockDim.x + threadIdx.x;
  if (id < n) {
    a[id] = exp(a[id]);
  }
}


__global__ void axpy_kernel(float *a, float *b, float scaleA,int lda,int ldy, int n, float *y){
  int id = (blockIdx.x + gridDim.x*blockIdx.y)*blockDim.x + threadIdx.x;
  if (id<n) {
       y[id*ldy] = scaleA*a[id*lda]+b[id*lda];
      //  printf("%.2f ",y[id*ldy]);
  }
}
