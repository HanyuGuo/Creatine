#ifndef _CUDACONVMATRIX_HPP_
#define _CUDACONVMATRIX_HPP_
#include "../include/im2col.cuh"
#include "../include/col2im.cuh"
#include "../include/cudaMatrix.cuh"
#include <cublas_v2.h>
#include <assert.h>

class cudaConvMatrix {
private:
  int _D0, _D1, _D2, _D3;
  int _numElements;
  float *devData;
  float *ret_data;
  void _init(float* data, int D0, int D1, int D2, int D3);

public:
  cudaConvMatrix();
  cudaConvMatrix(int D0, int D1, int D2, int D3);
  cudaConvMatrix(float* data, int D0, int D1, int D2, int D3);
  ~cudaConvMatrix();
  inline int getDim(int dim) const {
    if (dim == 0)
      return _D0;
    else if (dim == 1)
      return _D1;
    else if (dim == 2)
      return _D2;
    else if (dim == 3)
      return _D3;
    else{
      std::cout << "error input for getDim() in convMatrix\n"; 
      return 0; 
    }

  }

  inline void device2host() {
    cudaMemcpy(ret_data, devData, _numElements*sizeof(float), cudaMemcpyDeviceToHost);
  }



  inline float& getCell(int i, int j, int m, int n) {
    assert(i >= 0 && i < _D0);
    assert(j >= 0 && j < _D1);
    assert(m >= 0 && m < _D2);
    assert(n >= 0 && n < _D3);
    return ret_data[i*_D1*_D2*_D3 + j*_D2*_D3 + m*_D3 + n];
  }


  inline float& operator()(int i, int j,int m,int n){
    return getCell(i,j,m,n);
  }

  inline float *getDevData() const{
    if (devData != NULL)
    {
      return devData;
    } else{
       std::cout << "Data is not null!"<< "\n";
       exit(1);
    }

  }

  inline void getDeviceData(float *hdata){
    cudaError_t err; 
    cudaMemcpy(hdata,devData, _numElements*sizeof(float),cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    printf("%d\n", err);
  }
 
  void resize(cudaMatrix &tgt);

  void convolve(cudaConvMatrix &weights, bool samePadding, int stride, cudaConvMatrix &tgt);
  
};






#endif
