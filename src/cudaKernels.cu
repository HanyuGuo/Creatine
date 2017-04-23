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
    int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < n) {
    a[id] = pow(a[id],scale);
    id += gridDim.x*blockDim.x;
  }
}

__global__ void expgpu_kernel(float *a, int n){
  // int id = (blockIdx.x + gridDim.x*blockIdx.y)*blockDim.x + threadIdx.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < n) {
    a[id] = exp(a[id]);
    id +=   gridDim.x*blockDim.x;
  }
}


__global__ void axpy_kernel(float *a, float *b, float scaleA,int lda,int ldy, int n, float *y){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id<n) {
       y[id*ldy] = scaleA*a[id*lda]+b[id*lda];
      //  printf("%.2f ",y[id*ldy]);
  }
}

__global__ void add_mat_vec_kernel(float *a, float *b, int nr, int nc,float scale, float *y){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx < nc && idy < nr) {
        y[idy*nr+idx] = a[idy*nr+idx]+b[idx];
        // id +=   gridDim.x*blockDim.x;
    }

}

__global__ void DivideByScalar(float *x, float *scale, int nr, int nc){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  int idy = blockDim.y*blockIdx.y + threadIdx.y;
  if (idx < nc && idy < nr) {
    x[idy*nr+idx] = x[idy*nr+idx]/scale[idy];
  }

}
