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
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  while (id < n) {
    y[id] = exp(a[id]);
    
    id +=   gridDim.x*blockDim.x;
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


// __device__ void softmax_device(float *x, int n, int stride, float *y){
//    float sum = 0;
//    float largest = x[0];
//    for (int i = 0; i < n; ++i)
//    {
//      float val = x[i*stride];
//      largest = (val > largest)? val : largest;
//    }

//    for (int j = 0; j < n; ++j)
//    {
//       float exp_val = exp(x[j*stride]-largest);
//       sum += exp_val;
//       y[j*stride] = exp_val;
//    }

//    for (int k = 0; k < n; ++k)
//    {
//       y[k*stride] /= sum;
//    }


// }

// __global__ void softmax_kernel(float *x, int n, int nc, int stride, float *y) {
   
//    int id = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
//    if (id < n)
//    {
//     int offset = id+gridDim.x*blockDim*x;
//    softmax_device(x,nc,stride,y);
//     x = x+offset;
//     y= y+offset;

//    }
   

    
// }


__global__ void calc_max(int bs, int nc, float *x, float *max) {
   int index = blockIdx.x*blockDim.x + threadIdx.x;
   
   if (index < bs) {
      float maxval = x[index * nc];
      for (int i = index * nc + 1 ; i < (index + 1) * nc; i ++) {
        maxval = max(maxval, x[i]);
      }
      max[index] = maxval;
     
   }
     

}


__global__ void subtract_max(int bs, int nc, float *x, float *max, float *y){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  while (index < bs*nc)
  {
    y[index] = x[index] - max[index / nc];
    index += gridDim.x * blockDim.x;  

  }

}


__global__ void calc_sum_row(int bs, int nc, float *x, float *r_sum){
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (index < bs)
  {
    float sum = 0;
    for (int i = index*nc; i < (index+1)*nc; i++){
      sum += x[i];
    }

    r_sum[index] = sum;
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
