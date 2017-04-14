#include <iostream>
#include <cuda.h>
#include "include/cudaMatrix.hpp"
#include "include/Activations.cuh"

int main(int argc, char  *argv[]) {
  float *data1;
  float *data2;
  float *hdata1;
  float *hdata2;
  float *resdata;
  int numRows = 64, numCols = 64;
  int numelems = numRows*numCols;
  //std::cout<<"Num Elems:"<<numElems<<"\n";
  cudaMatrix cm1(numRows, numCols);
  cudaMatrix cm2(numRows, numCols);
  // cudaMatrix res(numRows,numCols);
  // data1 = new float[numRows*numCols];
  // data2 = new float[numRows*numCols];
  hdata1= new float[numelems];
  hdata2 = new float[numelems];
  // resdata = new float[numelems];
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; j++) {
         hdata1[i*numRows+j] = i;
        //  hdata2[i*numRows+j] = i;

    }
  }


  // std::cout<<"Setting device data..\n";
  // cm1.setDeviceData(data1,numelems);
  // cm2.setDeviceData(data2,numelems);
  // //cm1.cudaElemWiseDivide(cm2,res);
  // cm1.getDeviceData(hdata1);
  cudaMalloc((void**)&data1,numelems*sizeof(float));
  cudaMemcpy(data1,hdata1,numelems*sizeof(float),cudaMemcpyHostToDevice);
  Activation a = LOGISTIC;
  activations_on_gpu(data1,numelems,a);
  //cudaMemcpy(hdata2,data1,numelems*sizeof(float),cudaMemcpyDeviceToHost);
  // res.getDeviceData(hdata1);
  //
  // for (int i = 0; i < numRows; i++) {
  //     for (int j = 0; j < numCols; j++) {
  //         std::cout<<hdata2[i*numRows+numCols]<<"\n";
  //     }
  // }
  cudaFree(data1);
  return 0;
}
