#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_


__global__
void MatAddKernel(float *a, float *b, float *c, int numRows,int numCols){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  if(idy < numRows && idx < numCols) {
    c[idy*numRows+idx] = a[idy*numRows+idx]+b[idy*numRows+idx];
  }
}





#endif
