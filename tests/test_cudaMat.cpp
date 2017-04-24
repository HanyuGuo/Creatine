#include <iostream>
#include <cuda.h>
#include "include/cudaMatrix.hpp"
#include "include/Activations.cuh"

int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}
int main(int argc, char *argv[]) {
  float *data1;
  float *data2;
  float *hdata1;
  float *hdata2;
  float *resdata;
  int numRows1 = 2000, numCols1 = 10;
  int numRows2 = 200, numCols2 = 2000;
  // int numRows2 = 1, numCols2 = 3;
  // int numelems = numRows*numCols;
  //std::cout<<"Num Elems:"<<numElems<<"\n";

  data1 = new float [numRows1*numCols1];
  // data1 = new float[numRows1*numCols1];
  data2 = new float[numRows2*numCols2];
  hdata1= new float[numRows1*numCols1];
  int ResnumCols = 10;
  hdata2 = new float[numRows1*ResnumCols];
  // resdata = new float[numRows1*numCols1];
  for (int i = 0; i < numRows1; ++i) {
    for (int j = 0; j < numCols1; ++j) {
      // data1[i*numRows1 +j] = i+j;
      data1[ci(i,j,numCols1)] = j;

    }

  }
  // for (int i = 0; i < numRows1; ++i) {
  //   for (int j = 0; j < numCols1; j++) {
  //
  //        data1[i*numRows1+j] = i;
  //       //  data1[ci(i,j,numCols1)] = i;
  //       //  std::cout << i << " ";
  //       //  data2[i*numRows+j] = i;
  //
  //   }
  //   // std::cout << "\n";
  // }
  for (int i = 0; i < numRows2; ++i) {
    for (int j = 0; j < numCols2; j++) {
         data2[ci(i,j,numCols2)] = 1;
        //  data2[i*numRows2+j] = j;
        // std::cout << data2[ci(i,j,numCols2)] << " ";
    }
    // std::cout << "\n";
  }
  // cudaMatrix *cm1 = new cudaMatrix(data1, numRows1, numCols1);
  // cudaMatrix *cm2 = new cudaMatrix(data2, numRows2, numCols2);
  // cudaMatrix *res = new cudaMatrix(numRows1,numCols1);

   cudaMatrix cm1(data1,numRows1,numCols1);
   cudaMatrix cm2(data2, numRows2,numCols2);
   cudaMatrix res(numRows1,numCols2);
   cudaMatrix res1(numRows1,ResnumCols);
   cm1.softmax_gpu(res1);
  // cm1.calc_activation_gpu(RELU,res1);
  // cm1.gemm_ongpu(false,false, cm2,1,0,res);
  //  std::cout<<"Setting device data..\n";
  //  cm1.setDeviceData(data1,numRows1*numCols1);
  //  cm2.setDeviceData(data2,numelems);
  //  cm1.gemm_ongpu(false,false,cm2,1,0,res);
  // (*cm1).cudaAddv((*cm2),1,res);
  // cm1.axpy_ongpu(cm2,0.5,1,1,res);
    // cm1.powgpu(2);
  // cm1.expgpu();
   //cm1.getDeviceData(hdata1);
  //cm1.cudaElemWiseDivide(cm2,res);
  // res.getDeviceData(hdata1);
  // cudaMalloc((void**)&data1,numelems*sizeof(float));
  // cudaMemcpy(data1,hdata1,numelems*sizeof(float),cudaMemcpyHostToDevice);
  // Activation a = LOGISTIC;
  // activations_on_gpu(data1,numelems,a);
  //cudaMemcpy(hdata2,data1,numelems*sizeof(float),cudaMemcpyDeviceToHost);
  res1.getDeviceData(hdata2);
   // cm1.getDeviceData(hdata1);
  // res.getDeviceData(hdata1);
  
  for (int i = 0; i < numRows1; i++) {
    for (size_t j = 0; j < ResnumCols; j++) {
       std::cout<< hdata2[ci(i,j,ResnumCols)] << " ";
    }
   // std::cout<<"\n";
  }
  //cudaFree(data1);
  return 0;
}
