#ifndef _CUDACONVMATRIX_HPP_
#define _CUDACONVMATRIX_HPP_
#include "include/im2col.cuh"
#include <cublas_v2.h>

class cudaConvMatrix {
private:
   int _bs;
   int _height;
   int _width;
   int _stride;
   int _kern_sz;
   int _pad;
   int _ch_in;
   int n_elems;
   float *devData;
   void _init(float *data, int bs, int height, int width, int channels,int pad, int kern_sz, int stride);

public:
  cudaConvMatrix();
  cudaConvMatrix(float *data, int bs, int height, int width, int channels);
  cudaConvMatrix(int bs, int height, int width, int channels);
    ~ConvLayer();
   int getnDim(const int n) const{
    switch(n){
      case 1: return _height; 
              break;
      case 2: return _width;
              break;
      case 3: return _ch_in;
              break;
      case 4: return _bs;
              break;
    }
    return 0;
   }
float *getDevData() const{
  if (devData != NULL)
  {
    return devData;
  } else{
     std::cout << "Data is not null!"<< "\n";
     exit(1);
  }

}

void convgetData(float *hdata){
  cudaError_t err; 

  cudaMemcpy(hdata,devData, n_elems*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  printf("%d\n", err);
}
 

 void fwd_pass_convolution_gpu(cudaConvMatrix &weights,cudaConvMatrix &tgt);
 // void gemm_conv_gpu(bool tA, bool tB, const cudaConvMatrix &b, float scaleA, float scaleB, cudaConvMatrix &tgt, int offset); // for convolutions.

};






#endif
