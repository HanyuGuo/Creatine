#include <iostream>
#include <cuda.h>
#include "../include/cudaMatrix.hpp"

int main(int argc, char const *argv[]) {
  float *data1;
  float *data2;
  float *hdata1;
  float *hdata2;
  int numRows = 64, numCols = 64;
  int numElems = numRows*numCols;
  cudaMatrix cm1(numRows, numCols);
  cudaMatrix cm2(numRows, numCols);
  data1 = new float[numRows*numCols];
  data2 = new float[numRows*numCols];
  hdata1= new float[numElems];
  hdata2 = new float[numElems];
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; j++) {
         data1[i*numRows+j] = i;
         data2[i*numRows+j] = i;

    }
  }

  cm1.setDeviceData(data1,numElems);
  cm2.setDeviceData(data2,numElems);

  return 0;
}
