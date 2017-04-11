#include "../include/matrix.hpp"

void Matrix::_init(double* data, int64 numRows, int64 numCols, bool transpose) {
  _data = data;
  _numRows = numRows;
  _numCols = numCols;
  _numElements = _numRows * _numCols;
  _trans = transpose ? CblasTrans : CblasNoTrans;
}

Matrix::Matrix(){
  _init(NULL, 0, 0, false);
}

Matrix::Matrix(int64 numRows, int64 numCols) {
  _init(NULL, numRows, numCols, false);
  this->_data = numRows * numCols > 0 ? new double[this->_numElements] : NULL;
}

Matrix::Matrix(const Matrix &like) {
  _init(NULL, like.getNumRows(), like.getNumCols(), false);
  this->_data = new double[this->_numElements];
  for (int i = 0; i < _numElements; i++)
    this->_data[i] = 0;
}

Matrix::Matrix(double* data, int64 numRows, int64 numCols) {
  _init(data, numRows, numCols, false);
}


void Matrix::_elemWiseLoop(double (*func)(double), Matrix& target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j));
    }
  }
}

void Matrix::_elemWiseLoop(double (*func)(double), const double scale, Matrix& target) {
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)(scale * (*this)(i,j));
    }
  }
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

void Matrix::assign(double* data) {
  _data = data;
}

void Matrix::T() {
  if (_trans == CblasTrans)
    _trans = CblasNoTrans;
  else
    _trans = CblasTrans;
}

void Matrix::add(const Matrix &m) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_add, m, 1, *this);
}

void Matrix::add(const Matrix &m, const double scale) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_add, m, scale, *this);
}


// target = (this + m)*scale
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

void Matrix::subtracted(const double scale, Matrix &target) {
  Matrix tempM(target);
  _elemWiseLoop(&_add, -scale, tempM);
  tempM.dot(-1, target);
}

// target = this * scale
void Matrix::dot(const double scale, Matrix &target) {
  _elemWiseLoop(&_mul, scale, target);
}

// target = dot(this, m) * scale
void Matrix::dot(const Matrix &m, const double scale, Matrix &target) {
  _elemWiseLoop(&_mul, m, scale, target);
}

// target = dot(this, m)
void Matrix::dot(const Matrix &m, Matrix &target) {
  _elemWiseLoop(&_mul, m, target);
}


// target = scale * this * m
void Matrix::rightMult(const Matrix &m, const double scale, Matrix &target) {
  Product(*this, m, scale, false,  false, target);
}

// target = this * m
void Matrix::rightMult(const Matrix &m, Matrix &target) {
  rightMult(m, 1, target);
}

// target = scale * this * m (transpose)
void Matrix::rightMult(const Matrix &m, const double scale, Matrix &target, bool thisT, bool mT) {
  Product(*this, m, scale, thisT, mT, target);
}

void Matrix::rightMult(const Matrix &m, Matrix &target, bool thisT, bool mT) {
  rightMult(m, 1, target, thisT, mT);
}


void Matrix::rightMultPlus(const Matrix &m, const Matrix &p, Matrix &target) {
  Matrix tempM(target);
  rightMult(m, 1, tempM);
  tempM.add(p, target);

}


void Matrix::Product(const Matrix &a, const Matrix &b, double scaleAB, bool aT, bool bT, Matrix &c) {
  if (!aT && !bT) {
    assert(a.getNumCols() == b.getNumRows());
    assert(c.getNumRows() == a.getNumRows() && c.getNumCols() == b.getNumCols());
  }
  else if (aT && !bT) {
    assert(a.getNumRows() == b.getNumRows());
    assert(c.getNumRows() == a.getNumCols() && c.getNumCols() == b.getNumCols());
  }
  else if (!aT && bT) {
    assert(a.getNumCols() == b.getNumCols());
    assert(c.getNumRows() == a.getNumRows() && c.getNumCols() == b.getNumRows());
  }
  else {
    assert(a.getNumRows() == b.getNumCols());
    assert(c.getNumRows() == a.getNumCols() && c.getNumCols() == b.getNumRows());
  }
  CBLAS_TRANSPOSE aTrans = aT ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE bTrans = bT ? CblasTrans : CblasNoTrans;
  cblas_dgemm(CblasRowMajor, aTrans, bTrans, c._numRows, c._numCols, a._numCols, scaleAB, a._data,
    a._numCols, b._data, b._numCols, 1, c._data, c._numCols);
}

void Matrix::eltwiseDivideByScale(const double scale, Matrix &target) {
  _elemWiseLoop(&_divide, scale, target);
}

void Matrix::eltwiseScaleDivideByThis(const double scale, Matrix &target) {
  _elemWiseLoop(&_divided, scale, target);
}

void Matrix::exp(Matrix &target) {
  _elemWiseLoop(&_exp, target);
}

void Matrix::exp(const double scale, Matrix &target) {
  _elemWiseLoop(&_exp, scale, target);
}

void Matrix::ln(Matrix &target) {
  _elemWiseLoop(&_ln, target);
}

void Matrix::log(Matrix &target) {
  _elemWiseLoop(&_log, target);
}

void Matrix::reduce_sum(double &sum) {
  sum = 0;
  for (int64 i=0; i < getNumRows(); i++) {
    for (int64 j=0; j < getNumCols(); j++) {
      sum = sum + (*this)(i,j);
    }
  }
}


Matrix::~Matrix() {
	if(this->_data != NULL) {
		delete[] this-> _data ;
	}
}