#ifndef CONVMATRIX_H
#define CONVMATRIX_H
#include "assert.h"
#include "basic_calc.hpp"
#include "matrix.hpp"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream> //for debugging
#include "../include/im2col.cuh"
extern "C" {
  #include <cblas.h>
}


class convMatrix {
private:
  float* _data;
  int _D0, _D1, _D2, _D3;
  int _numElements;
  void _init(float* data, int D0, int D1, int D2, int D3);
public:
  convMatrix();
  convMatrix(int D0, int D1, int D2, int D3);
  convMatrix(float* data, int D0, int D1, int D2, int D3);
  inline float& getCell(int i, int j, int m, int n) {
    assert(i >= 0 && i < _D0);
    assert(j >= 0 && j < _D1);
    assert(m >= 0 && m < _D2);
    assert(n >= 0 && n < _D3);
    return _data[i*_D1*_D2*_D3 + j*_D2*_D3 + m*_D3 + n];
  }
  float& operator()(int i, int j, int m, int n) {
    return getCell(i, j, m, n);
  }
  inline float* getData() {
    return _data;
  }
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
  inline int getNumElements() const {
    return _numElements;
  }
  void assign(float* data);
  void print_data();
  void convolve(convMatrix &filter, int stride, bool samePadding,  convMatrix &target);
  void flatten(Matrix &target);
  void fwdpassconvgpu(const convMatrix &weights, convMatrix col,int bs, int channels, int height, int width, int stride, int padding, int kern_sz, int col_height, int col_width, convMatrix &tgt); // calc fwd pass gpu.
};

#endif
