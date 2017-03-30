#include "../include/matrix.hpp"

void Matrix::_init(double* data, int64 numRows, int64 numCols) {
  _data = data;
  _numRows = numRows;
  _numCols = numCols;
  _numElements = _numRows * _numCols;
}

Matrix::Matrix(){
  _init(NULL, 0, 0);
}

Matrix::Matrix(int64 numRows, int64 numCols) {
  _init(NULL, numRows, numCols);
  this->_data = numRows * numCols > 0 ? new double[this->_numElements] : NULL;
}

Matrix::Matrix(const Matrix &like) {
  _init(NULL, like.getNumRows(), like.getNumCols());
  this->_data = new double[this->_numElements];

}

Matrix::Matrix(double* data, int64 numRows, int64 numCols) {
  _init(data, numRows, numCols);
}

void Matrix::_elemWiseLoop(double (*func)(double, double), const double scale, Matrix &target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), scale);
    }
  }
}

void Matrix::_elemWiseLoop(double (*func)(double, double), const Matrix &m, Matrix &target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(i,j));
    }
  }
}

void Matrix::_elemWiseLoop(double (*func)(double, double), const Matrix &m, const double scale, Matrix &target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(i,j)) * scale;
    }
  }
}

void Matrix::_elemWiseLoop(double (*func)(double, double, double), const Matrix &m, const double scale, Matrix &target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(i,j), scale);
    }
  }
}

void Matrix::add(const Matrix &m, const double scale, Matrix &target) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_add, m, scale, target);
}

void Matrix::add(const double scale, Matrix &target) {
  _elemWiseLoop(&_add, scale, target);
}

void Matrix::add(const Matrix &m, Matrix &target) {
  add(m, 1, target);
}

void Matrix::subtract(const Matrix &m, Matrix &target) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_addScaleY, m, -1, target);
}

void Matrix::subtract(const double scale, Matrix &target) {
  _elemWiseLoop(&_add, -scale, target);
}


// target = scale * this * m
void Matrix::rightMult(const Matrix &m, const double scale, Matrix &target) {
  Product(*this, m, scale, target);
}


// target = this * m
void Matrix::rightMult(const Matrix &m, Matrix &target) {
  rightMult(m, 1, target);
}

void Matrix::Product(const Matrix &a, const Matrix &b, double scaleAB, Matrix &c) {
  assert(a.getNumCols() == b.getNumRows());
  assert(c.getNumRows() == a.getNumRows() && c.getNumCols() == b.getNumCols());
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a._numRows, b._numCols, a._numCols, scaleAB, a._data,
    a._numCols, b._data, b._numCols, 1, c._data, c._numCols);
}


Matrix::~Matrix() {
	if(this->_data != NULL) {
		delete[] this-> _data ;
	}
}