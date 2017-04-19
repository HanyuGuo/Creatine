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
extern "C" {
  #include <cblas.h>
}

typedef long long int int64;

class convMatrix {
private:
  float* _data;
  int64 _D0, _D1, _D2, _D3;
  int64 _numElements;
  void _init(float* data, int64 D0, int64 D1, int64 D2, int64 D3);
public:
  convMatrix();
  convMatrix(int64 D0, int64 D1, int64 D2, int64 D3);
  convMatrix(float* data, int64 D0, int64 D1, int64 D2, int64 D3);
  inline float& getCell(int64 i, int64 j, int64 m, int64 n) {
    assert(i >= 0 && i < _D0);
    assert(j >= 0 && j < _D1);
    assert(m >= 0 && m < _D2);
    assert(n >= 0 && n < _D3);
    return _data[i*_D1*_D2*_D3 + j*_D2*_D3 + m*_D3 + n];
  }
  float& operator()(int64 i, int64 j, int64 m, int64 n) {
    return getCell(i, j, m, n);
  }
  inline float* getData() {
    return _data;
  }
  inline int64 getDim(int64 dim) const {
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
  inline int64 getNumElements() const {
    return _numElements;
  }
  void assign(float* data);
  void print_data();
  void convolve(convMatrix &filter, int64 stride, bool samePadding,  convMatrix &target);
  void flatten(Matrix &target);
};

#endif