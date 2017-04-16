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

typedef long long int int64;

class Matrix {
private:
  double* _data;
  int64 _numRows, _numCols;
  int64 _numElements;
  CBLAS_TRANSPOSE _trans;
  void _init(double* data, int64 numRows, int64 numCols, bool transpose);
  void _elemWiseLoop(double (*func)(double), Matrix& target);
  void _elemWiseLoop(double (*func)(double), const double scale, Matrix& target);
  void _elemWiseLoop(double (*func)(double, double), const double scale, Matrix& target);
  void _elemWiseLoop(double (*func)(double, double), const Matrix &m, Matrix& target);
  void _elemWiseLoop(double (*func)(double, double), const Matrix &m, const double scale, Matrix& target);
  void _elemWiseLoop(double (*func)(double, double, double), const Matrix &m, const double scale, Matrix &target);
public:
  Matrix();
  Matrix(int64 numRows, int64 numCols);
  Matrix(const Matrix &like);
  Matrix(double* data, int64 numRows, int64 numCols);

  inline double& getCell(int64 i, int64 j) const {
    assert(i >= 0 && i < _numRows);
    assert(j >= 0 && j < _numCols);
    return _data[i * _numCols + j];
  }
  double& operator()(int64 i, int64 j) const {
    return getCell(i, j);
  }
  inline double* getData() const {
    return _data;
  }
  inline int64 getNumRows() const {
    return _numRows;
  }
  inline int64 getNumCols() const {
    return _numCols;
  }
  inline int64 getNumElements() const {
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
  void assign(double* data);
  void copyTo(Matrix &m);
  void T();
  void add(const Matrix &m);
  void add(const Matrix &m, const double scale);
  void add(const Matrix &m, const double scale, Matrix &target);  //basic addition
  void add(const double scale, Matrix &target);
  void add(const Matrix &m, Matrix &target);
  void subtract(const Matrix &m, Matrix &target);
  void subtract(const double scale, Matrix &target);
  void subtracted(const double scale, Matrix &target);
  void Product(const Matrix &a, const Matrix &b, double scaleAB, bool aT, bool bT, Matrix &c);
  void dot(const double scale, Matrix &target);
  void dot(const Matrix &m, const double scale, Matrix &target);
  void dot(const Matrix &m, Matrix &target);
  void rightMult(const Matrix &m, const double scale, Matrix &target);
  void rightMult(const Matrix &m, Matrix &target);
  void rightMult(const Matrix &m, const double scale, Matrix &target, bool thisT, bool mT);
  void rightMult(const Matrix &m, Matrix &target, bool thisT, bool mT);
  void rightMultPlus(const Matrix &m, const Matrix &p, Matrix &target);
  void eltwiseDivideByScale(const double scale, Matrix &target);
  void eltwiseDivide(const Matrix &m, Matrix &target);
  void eltwiseScaleDivideByThis(const double scale, Matrix &target);
  void exp(Matrix &target);
  void exp(const double scale, Matrix &target);
  void ln(Matrix &target);
  void log(Matrix &target);
  void reduce_sum(Matrix &sum); // only do all to one currently
  double max() const;
  void max(const double scale, Matrix &target);
  void reluGrads(const Matrix &m, Matrix &target);
  void sigmoidGrads(const Matrix &m,  Matrix &target);
  void argmax(double * result);

  ~Matrix();
};
#endif