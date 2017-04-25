#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
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

__global__ void expgpu_kernel(float *a, int n, float* y){
  // int id = (blockIdx.x + gridDim.x*blockIdx.y)*blockDim.x + threadIdx.x;
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  while (id < n) {
    y[id] = exp(a[id]);
    
    id +=  gridDim.x*blockDim.x;
  }
}


__global__ void axpy_kernel(float *a, float *b, float scaleA,int lda,int ldy, int n, float *y){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id<n) {
       y[id*ldy] = scaleA*a[id*lda]+b[id*lda];
      //  printf("%.2f ",y[id*ldy]);
       id += gridDim.x*blockDim.x;
  }
}

__global__ void add_mat_vec_kernel(float *a, float *b, int nr, int nc, float *y){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    while (idx < nc * nr) {
        y[idx] = a[idx]+b[idx%nc];
        idx += gridDim.x*blockDim.x;
    }

}

__global__ void DivideByScalar(float *x, float *scale, int nr, int nc, float* y){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  int idy = blockDim.y*blockIdx.y + threadIdx.y;
  if (idx < nc && idy < nr) {
    y[idy*nr+idx] = x[idy*nr+idx]/scale[idy];
  }

}




__global__ void calc_max(int bs, int nc, float *x, float *max) {
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  while (index < bs) {
    float maxval = x[index * nc];
    for (int i = index * nc + 1 ; i < (index + 1) * nc; i ++) {
      maxval = max(maxval, x[i]);
    }
    max[index] = maxval;
    index += blockDim.x * gridDim.x;
  }
     

}


__global__ void subtract_max(int bs, int nc, float *x, float *max, float *y){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < bs*nc)
  {
    y[index] = x[index] - max[index / nc];
    // if (index > 500)
    // printf("%d ",gridDim.x);
    index += gridDim.x * blockDim.x;  
  }

}


__global__ void calc_sum_row(int bs, int nc, float *x, float *r_sum){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  while (index < bs)
  {
    float sum = 0;
    for (int i = index*nc; i < (index+1)*nc; i++){
      sum += x[i];
    }

    r_sum[index] = sum;
    index += gridDim.x * blockDim.x;  
  }
}


__global__ void div_row(int bs, int nc,float *x, float *row, float *y){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  while (index < bs*nc)
  {
    y[index] = x[index]/row[index / nc];
    index += gridDim.x * blockDim.x;  

  }

}


__global__ void argmax(int bs, int nc, float *x, int *y){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  while (index < bs)
  {
    int largest_idx = 0;
    float largest = x[index*nc];
    for (int i = index*nc + 1; i < (index+1)*nc; i++){
      if (largest < x[i]){
        largest = x[i];
        largest_idx = i - index*nc;
      }
    }

    y[index] = largest_idx;
    index += gridDim.x * blockDim.x;  
  }
}