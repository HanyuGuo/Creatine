#include <iostream>
#include <cuda.h>
#include "../include/cudaMatrix.cuh"


int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}
int main(int argc, char *argv[]) {
  float *data1;
  float *data2;
  float *hdata1;
  float *resdata;
  int numRows1 = 200, numCols1 = 100;
  int numRows2 = 100, numCols2 = 50;
  // int numRows2 = 1, numCols2 = 3;
  // int numelems = numRows*numCols;
  //std::cout<<"Num Elems:"<<numElems<<"\n";

  data1 = new float [numRows1*numCols1];
  // data1 = new float[numRows1*numCols1];
  data2 = new float[numRows2*numCols2];
  hdata1= new float[numRows1*numCols2];

  // resdata = new float[numRows1*numCols1];
  for (int i = 0; i < numRows1; ++i) {
    for (int j = 0; j < numCols1; ++j) {
      data1[ci(i,j,numCols1)] = i+j;
      std::cout << data1[ci(i,j,numCols1)] << " ";
    }
    std::cout << "\n";
  }

  for (int i = 0; i < numRows2; ++i) {
    for (int j = 0; j < numCols2; j++) {
         data2[ci(i,j,numCols2)] = i+j;
         std::cout << data2[ci(i,j,numCols2)] << " ";
    }
    std::cout << "\n";
  }

  // for (int i = 0; i < numRows1; ++i) {
  //   for (int j = 0; j < numCols2; j++) {
  //        hdata1[ci(i,j,numCols2)] = 0;

  //   };
  // }


   cudaMatrix cm1(data1,numRows1,numCols1);
   cudaMatrix cm2(data2, numRows2,numCols2);
   cudaMatrix res(numRows1,numCols2);


  cm1.gemm_ongpu(false, false, cm2, 1, 0, res);
   // cm1.getDeviceData(hdata1);
  res.getDeviceData(hdata1);
  // std::cout << "Matrix dimensions are not same .. aborting \n";
  for (int i = 0; i < numRows1; i++) {
    for (size_t j = 0; j < numCols2; j++) {
       std::cout<< hdata1[ci(i,j,numCols2)] << " ";
    }
   std::cout<<"\n";
  }
  //cudaFree(data1);
  return 0;
}
