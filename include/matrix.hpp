#ifndef MATRIX_H
#define MATRIX_H
#include "assert.h"
#include "basic_calc.hpp"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream> //for debugging
extern "C" {
  #include <cblas.h>
}


class Matrix {
private:
  float* _data;
  int _numRows, _numCols;
  int _numElements;
  CBLAS_TRANSPOSE _trans;
  void _init(float* data, int numRows, int numCols, bool transpose);
  void _elemWiseLoop(float (*func)(float), Matrix& target);
  void _elemWiseLoop(float (*func)(float), const float scale, Matrix& target);
  void _elemWiseLoop(float (*func)(float, float), const float scale, Matrix& target);
  void _elemWiseLoop(float (*func)(float, float), const Matrix &m, Matrix& target);
  void _elemWiseLoop(float (*func)(float, float), const Matrix &m, const float scale, Matrix& target);
  void _elemWiseLoop(float (*func)(float, float, float), const Matrix &m, const float scale, Matrix &target);
public:
  Matrix();
  Matrix(int numRows, int numCols);
  Matrix(const Matrix &like);
  Matrix(float* data, int numRows, int numCols);



  inline float& getCell(int i, int j) const {
    assert(i >= 0 && i < _numRows);
    assert(j >= 0 && j < _numCols);
    return _data[i * _numCols + j];
  }
  float& operator()(int i, int j) const {
    return getCell(i, j);
  }
  inline float* getData() const {
    return _data;
  }
  inline int getNumRows() const {
    return _numRows;
  }
  inline int getNumCols() const {
    return _numCols;
  }
  inline int getNumElements() const {
    return _numElements;
  }
  inline bool sameDim(const Matrix &m) const {
    if (m.getNumRows() == getNumRows() && m.getNumCols() == getNumCols())
      return true;
    else 
      return false;
  }
  inline bool istrans() const {
    return _trans == CblasTrans;
  }
  void assign(float* data);
  void copyTo(Matrix &m);
  void T();
  void add(const Matrix &m);
  void add(const Matrix &m, const float scale);
  void add(const Matrix &m, const float scale, Matrix &target);  //basic addition
  void add(const float scale, Matrix &target);
  void add(const Matrix &m, Matrix &target);
  void subtract(const Matrix &m, Matrix &target);
  void subtract(const float scale, Matrix &target);
  void subtracted(const float scale, Matrix &target);
  void Product(const Matrix &a, const Matrix &b, float scaleAB, bool aT, bool bT, Matrix &c);
  void dot(const float scale, Matrix &target);
  void dot(const Matrix &m, const float scale, Matrix &target);
  void dot(const Matrix &m, Matrix &target);
  void rightMult(const Matrix &m, const float scale, Matrix &target);
  void rightMult(const Matrix &m, Matrix &target);
  void rightMult(const Matrix &m, const float scale, Matrix &target, bool thisT, bool mT);
  void rightMult(const Matrix &m, Matrix &target, bool thisT, bool mT);
  void rightMultPlus(const Matrix &m, const Matrix &p, Matrix &target);
  void eltwiseDivideByScale(const float scale, Matrix &target);
  void eltwiseDivide(const Matrix &m, Matrix &target);
  void eltwiseScaleDivideByThis(const float scale, Matrix &target);
  void exp(Matrix &target);
  void exp(const float scale, Matrix &target);
  void ln(Matrix &target);
  void log(Matrix &target);
  void reduce_sum(Matrix &sum); // only do all to one currently
  float max() const;
  void max(const float scale, Matrix &target);
  void reluGrads(const Matrix &m, Matrix &target);
  void sigmoidGrads(const Matrix &m,  Matrix &target);
  void argmax(float * result);

  ~Matrix();
};
#endif