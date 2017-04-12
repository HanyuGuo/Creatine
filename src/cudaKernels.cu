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
