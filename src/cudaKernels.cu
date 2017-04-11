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

void LaunchAddKernel(float *a, float *b, float *c, int numRows, int numCols) {
  cudaError_t err;
  // if (this->numElems != b.numElems && this->numElems != c.numElems) {
  //     std::cout<<" Matrix addition is not possible with different dims";
  // }
  int block_dim_x = 32;
  int block_dim_y = 32;
  int grid_dim_x = (this->numElems)/block_dim_x;
  int grid_dim_y = (this->numElems)/block_dim_y;
  dim3 grid(grid_dim_x,grid_dim_y, 1);
  dim3 block(block_dim_x, block_dim_y);
  std::cout << "Launching kernel now..." << '\n';
  MatAddKernel<<< grid, block >>>(this->devData,b.getDevData(),c.getDevData(), this->numRows, this->numCols);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't write to device data." << '\n';
  }
}
