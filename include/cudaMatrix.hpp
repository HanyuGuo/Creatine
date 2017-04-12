#ifndef _CUDA_MATRIX_HPP_
#define _CUDA_MATRIX_HPP_
#include<iostream>
#include "../include/cudaKernels.cuh"

class cudaMatrix {
private:
  float *devData;
  int numCols;
  int numRows;
  int numElems;
  void _init(float *data, int numRows, int numCols);

public:
  cudaMatrix(int numRows, int numCols);
  cudaMatrix(float *data, int numRows, int numCols);
  virtual ~cudaMatrix();
  int getNumRows() const {return numRows;}
  int getnumCols() const {return numCols;}
  void setDeviceData(float *data, int elems); // set device Data;
  void getDeviceData(float *hdata); // get device data in host pointer.
  float * getDevData() const {
     if (devData != NULL) {
       return devData;
     } else {
       std::cout << "Aborting...\n";
       exit(1);
     }

  }
  __host__ __device__ void cudaAdd(const cudaMatrix &b, cudaMatrix &c); // Matrix addition kernel.
  //void cudaAdd(const cudaMatrix &b);
};



#endif
