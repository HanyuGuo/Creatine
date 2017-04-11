#include <iostream>
#include <assert.h>
#include <cuda.h>
#include<cuda_helper.h>
#include "../include/cudaMatrix.hpp"


#define GPU


void cudaMatrix::_init(float *data, int numRows, int numCols){
  devData = data;
  numRows = numRows;
  numCols = numCols;
  numElems = numRows * numCols;
  cudaError_t err;
  if (numElems > 0 && devData != NULL) {
     cudaMalloc((void**)&devData, numElems*sizeof(float));
     err = cudaGetLastError();
     if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
     }
  }
}

cudaMatrix::cudaMatrix(int numRows, int numCols){
  cudaError_t err;
  _init(NULL,numRows,numCols);
  if (numRows*numCols > 0) {
    cudaMalloc((void**)&devData, numElems*sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Couldn't allocate memory\n";
    }
  }
}


cudaMatrix::cudaMatrix(float *data, int numRows, int numCols) {
   _init(data,numRows,numCols);
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
  cudaMemcpy(devData, data,elems*sizeof(float),cudaMemcpyHostToDevice);
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
  cudaMemcpy(devData, hdata, numElems*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't write to device data." << '\n';
  }
}



void cudaMatrix::cudaAdd(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (this->numRows == b.getNumRows() && this->numCols == b.getnumCols()) {
     LaunchAddKernel(devData, b.getDevData(), c.getDevData(), this->numRows, this->numCols);
  } else {
    std::cout<<"Matrix dims must be same, aborting..\n";
    exit(1);
  }

}
