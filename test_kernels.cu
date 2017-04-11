#include<cuda.h>
#include<stdio.h>
#include "include/cudaKernels.cuh"

int main(int argc, char **argv) {
  cudaError_t err;
  float *hdata1, *hdata2,*reshdata;
  float *ddata1, *ddata2,*resddata;
  int numRows, numCols,i,j;
  numRows  = numCols = 64;
  hdata1 = (float *) malloc(numRows*numCols*sizeof(float));
  hdata2 = (float *) malloc(numRows*numCols*sizeof(float));
  reshdata = (float *) malloc(numRows*numCols*sizeof(float));

  cudaMalloc((void**)&ddata1,numRows*numCols*sizeof(float));
  cudaMalloc((void**)&ddata2,numRows*numCols*sizeof(float));
  cudaMalloc((void**)&resddata,numRows*numCols*sizeof(float));
  for(i=0; i<numRows; ++i) {
    for(j=0; j<numCols; ++j) {
       hdata1[i*numRows+j] = i;
       hdata2[i*numRows+j] = i;
    }
  }
  cudaMemcpy(ddata1,hdata1, numRows*numCols*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(ddata2,hdata2, numRows*numCols*sizeof(float),cudaMemcpyHostToDevice);
  int block_dim_x = 32;
  int block_dim_y = 32;
  int grid_dim_x = (numRows*numCols)/block_dim_x;
  int grid_dim_y = (numRows*numCols)/block_dim_y;
  dim3 grid(grid_dim_x,grid_dim_y, 1);
  dim3 block(block_dim_x, block_dim_y);
  printf("Launching kernel now...\n");
  MatAddKernel<<< grid, block >>>(ddata1,ddata2,resddata, numRows, numCols);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error launching kernel\n");
  }
  cudaMemcpy(reshdata,resddata,numRows*numCols*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Can't read from device data \n");
  }

  return 0;
}
