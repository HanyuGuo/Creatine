#include <iostream>
#include "include/cudaMatrix.hpp"

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
  cudaMatrix res(numRows,numCols);
  data1 = new float[numRows*numCols];
  data2 = new float[numRows*numCols];
  hdata1= new float[numelems];
  hdata2 = new float[numelems];
  resdata = new float[numelems];
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; j++) {
         data1[i*numRows+j] = i;
         data2[i*numRows+j] = i;

    }
  }


  std::cout<<"Setting device data..\n";
  cm1.setDeviceData(data1,numelems);
  cm2.setDeviceData(data2,numelems);
  cm1.cudaElemWiseDivide(cm2,res);
  res.getDeviceData(hdata1);
  for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
          std::cout<<hdata1[i*numRows+numCols]<<"\n";
      }
  }
  return 0;
}
