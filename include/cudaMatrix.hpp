#ifndef _CUDA_MATRIX_HPP_
#define _CUDA_MATRIX_HPP_
#include<iostream>
#include <vector>
#include<algorithm>
#include "../include/cudaKernels.cuh"
#include "../include/Activations.cuh"

class cudaMatrix {
private:
  float *devData;
  int numCols;
  int numRows;
  int numElems;
  int gpuid;
  void _init(float *data, int numRows, int numCols);
  void setbestcudaDevice();
  int ci(int row, int column, int nColumns) {
    return row*nColumns+column;
  }
public:
  cudaMatrix(int numRows, int numCols);
  cudaMatrix(float *data, int numRows, int numCols);
  virtual ~cudaMatrix();
  int getNumRows() const {return numRows;}
  int getNumCols() const {return numCols;}
  int getNumElems() const {return numElems;}
  int getLeadingDim() const {return numCols;}
  int getFollowingDim() const {return numRows;}
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
  void cudaAddv(const cudaMatrix &b, float scale, cudaMatrix &tgt);
  // float * reshape_data(float *data, int numCols, int numRows);
  void cudaAdd(const cudaMatrix &b, cudaMatrix &c); // Matrix addition kernel.
  void cudaWeightedAdd(const cudaMatrix &b,cudaMatrix &c,float scale); // WeightedAdd kernel.
  void cudaElemWiseMult(const cudaMatrix &b, cudaMatrix &c);
  void cudaElemWiseDivide(const cudaMatrix &b, cudaMatrix &c);
  void powgpu(int scale, int n);
  void expgpu(int n);
  void axpy_ongpu(const cudaMatrix &b, float scaleA, int ldx, int ldy, cudaMatrix &tgt); // perform axpy by striding the vector in a column major format
  void axpy_ongpu(const cudaMatrix &b, float scaleA, int ldx, int ldy);
  void gemm_ongpu(bool tA, bool tB, const cudaMatrix &b, float scaleA, float scaleB, cudaMatrix &tgt); // Sgemm on GPU.
  void calc_activation_gpu(Activation a, cudaMatrix &tgt);
  void softmax_gpu(cudaMatrix &tgt);
};



#endif
