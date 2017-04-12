#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include "../include/cudaMatrix.hpp"


#define GPU


void cudaMatrix::_init(float *data, int numrows, int numcols){
  devData = data;
  numRows = numrows;
  numCols = numcols;
  numElems = numRows * numCols;
  cudaError_t err;

  if (devData != NULL) {
    std::cout << "got data!";
  }
  if (numElems > 0 && devData != NULL) {
     cudaMalloc((void**)&devData, numElems*sizeof(float));
     err = cudaGetLastError();
     if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
     }
  }
}

cudaMatrix::cudaMatrix(int numrows, int numcols){
  cudaError_t err;
  _init(NULL,numrows,numcols);
  if (numRows*numCols > 0) {
    cudaMalloc((void**)&devData, numElems*sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Couldn't allocate memory\n";
    }
  }
}


cudaMatrix::cudaMatrix(float *data, int numrows, int numcols){
   _init(data,numrows,numcols);
}


cudaMatrix::~cudaMatrix(){
  cudaError_t err;
  if (numElems > 0 ) {
      cudaFree(devData);
      err = cudaGetLastError();

      if (err != cudaSuccess) {
        std::cout << "Can't free memory\n";
      }
  }
}

void cudaMatrix::setDeviceData(float *data, int elems) {
  cudaError_t err;

  if (elems != numElems) {
    std::cout<< "The size of data must be same! Aborting..\n";
    exit(1);
  }

  cudaMemcpy(devData,data,elems*sizeof(float),cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't write to device data." << '\n';
  }
}


void cudaMatrix::getDeviceData(float *hdata) {
  cudaError_t err;
  // if (hdata == NULL) {
  //    *hdata = new float[numElems];
  // }
  cudaMemcpy(hdata,devData, numElems*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't read device data." << '\n';
  }
}



 void cudaMatrix::cudaAdd(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x,grid_dim_y, 1);
    dim3 block(block_dim_x, block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout<<"Launching kernel now...\n";
    MatAddKernel<<<grid, block>>>(adata,bdata,resdata,numRows, numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }
  } else {
    printf("Matrix dims must be same, aborting..\n");
    exit(1);
  }

}


void cudaMatrix::cudaWeightedAdd(const cudaMatrix &b, cudaMatrix &c, float scale) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x,grid_dim_y, 1);
    dim3 block(block_dim_x, block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout << "Launching Weighted Add..." << '\n';
    WeightedAddKernel<<< grid, block >>>(adata,bdata,resdata,scale,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }
  } else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}



void cudaMatrix::cudaElemWiseMult(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x, grid_dim_y,1);
    dim3 block(block_dim_x,block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout<<"Doing Elt wise mat mul..."<<"\n";
    EltWiseMatMul<<< grid, block >>> (adata, bdata, resdata,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }

  }else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}


void cudaMatrix::cudaElemWiseDivide(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x, grid_dim_y,1);
    dim3 block(block_dim_x,block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout<<"Doing Elt wise mat divide..."<<"\n";
    EltWiseMatDivide<<< grid, block >>> (adata, bdata, resdata,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }

  }else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}
