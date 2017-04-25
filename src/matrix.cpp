#include "../include/matrix.hpp"

void Matrix::_init(float* data, int numRows, int numCols, bool transpose) {
  _data = data;
  _numRows = numRows;
  _numCols = numCols;
  _numElements = _numRows * _numCols;
  _trans = transpose ? CblasTrans : CblasNoTrans;
}


Matrix::Matrix(){
  _init(NULL, 0, 0, false);
}

Matrix::Matrix(int numRows, int numCols) {
  _init(NULL, numRows, numCols, false);
  this->_data = numRows * numCols > 0 ? new float[this->_numElements] : NULL;
}

Matrix::Matrix(const Matrix &like) {
  _init(NULL, like.getNumRows(), like.getNumCols(), false);
  this->_data = new float[this->_numElements];
  for (int i = 0; i < _numElements; i++)
    this->_data[i] = 0;
}

Matrix::Matrix(float* data, int numRows, int numCols) {
  _init(data, numRows, numCols, false);
}




void Matrix::_elemWiseLoop(float (*func)(float), Matrix& target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j));
    }
  }
}

void Matrix::_elemWiseLoop(float (*func)(float), const float scale, Matrix& target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)(scale * (*this)(i,j));
    }
  }
}


void Matrix::_elemWiseLoop(float (*func)(float, float), const float scale, Matrix &target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), scale);
    }
  }
}

void Matrix::_elemWiseLoop(float (*func)(float, float), const Matrix &m, Matrix &target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(i,j));
    }
  }
}

void Matrix::_elemWiseLoop(float (*func)(float, float), const Matrix &m, const float scale, Matrix &target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(0,j)) * scale;
    }
  }
}

void Matrix::_elemWiseLoop(float (*func)(float, float, float), const Matrix &m, const float scale, Matrix &target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i, j) = (*func)((*this)(i,j), m(i,j), scale);
    }
  }
}

void Matrix::assign(float* data) {
  _data = data;
}

void Matrix::copyTo(Matrix &m) {
  assert(getNumRows() == m.getNumRows());
  assert(getNumCols() == m.getNumCols());
  _elemWiseLoop(&_identity, m);
}

void Matrix::T() {
  if (_trans == CblasTrans)
    _trans = CblasNoTrans;
  else
    _trans = CblasTrans;
}

void Matrix::add(const Matrix &m) {
  assert(this->sameDim(m));
  add(m, 0);
}

void Matrix::add(const Matrix &m, const float scale) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_addScaleY, m, scale, *this);
}


// target = (this + m)*scale
void Matrix::add(const Matrix &m, const float scale, Matrix &target) {
  // assert(this->sameDim(m));
  _elemWiseLoop(&_add, m, scale, target);
}

void Matrix::add(const float scale, Matrix &target) {
  _elemWiseLoop(&_add, scale, target);
}

void Matrix::add(const Matrix &m, Matrix &target) {
  add(m, 1, target);
}

void Matrix::subtract(const Matrix &m, Matrix &target) {
  assert(this->sameDim(m));
  _elemWiseLoop(&_addScaleY, m, -1, target);
}

void Matrix::subtract(const float scale, Matrix &target) {
  _elemWiseLoop(&_add, -scale, target);
}

void Matrix::subtracted(const float scale, Matrix &target) {
  Matrix tempM(target);
  _elemWiseLoop(&_add, -scale, tempM);
  tempM.dot(-1, target);
}

// target = this * scale
void Matrix::dot(const float scale, Matrix &target) {
  _elemWiseLoop(&_mul, scale, target);
}

// target = dot(this, m) * scale
void Matrix::dot(const Matrix &m, const float scale, Matrix &target) {
  _elemWiseLoop(&_mul, m, scale, target);
}

// target = dot(this, m)
void Matrix::dot(const Matrix &m, Matrix &target) {
  _elemWiseLoop(&_mul, m, target);
}


// target = scale * this * m
void Matrix::rightMult(const Matrix &m, const float scale, Matrix &target) {
  Product(*this, m, scale, false,  false, target);
}

// target = this * m
void Matrix::rightMult(const Matrix &m, Matrix &target) {
  rightMult(m, 1, target);
}

// target = scale * this * m (transpose)
void Matrix::rightMult(const Matrix &m, const float scale, Matrix &target, bool thisT, bool mT) {
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


void Matrix::Product(const Matrix &a, const Matrix &b, float scaleAB, bool aT, bool bT, Matrix &c) {

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
  
  for (int i = 0; i < c.getNumRows(); i++) {
    for (int j = 0; j < c.getNumCols(); j++) {
      c(i,j) = 0;
    }
  }

  CBLAS_TRANSPOSE aTrans = aT ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE bTrans = bT ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, aTrans, bTrans, c.getNumRows(), c.getNumCols(), a.getNumCols(), scaleAB, a.getData(),
    a.getNumCols(), b.getData(), b.getNumCols(), 1, c.getData(), c.getNumCols());

}

void Matrix::eltwiseDivideByScale(const float scale, Matrix &target) {
  _elemWiseLoop(&_divide, scale, target);
}

void Matrix::eltwiseDivide(const Matrix &m, Matrix &target) {
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      target(i,j) = (*this)(i,j) / m(i,0);
    }
  }
}

void Matrix::eltwiseScaleDivideByThis(const float scale, Matrix &target) {
  _elemWiseLoop(&_divided, scale, target);
}

void Matrix::exp(Matrix &target) {

  Matrix tempM(target);
  subtract(max(), tempM);

  tempM._elemWiseLoop(&_exp, target);
}

void Matrix::exp(const float scale, Matrix &target) {
  // using namespace std;
  // cout << max() << endl;
  // Matrix tempM(target);
  // subtract(max(), tempM);
  // cout << (*this)(0,0)<<tempM(0,0) <<endl;
  _elemWiseLoop(&_exp, scale, target);
}

void Matrix::ln(Matrix &target) {
  _elemWiseLoop(&_ln, target);
}

void Matrix::log(Matrix &target) {
  _elemWiseLoop(&_log, target);
}

void Matrix::reduce_sum(Matrix &sum) {
  for (int i=0; i < getNumRows(); i++) {
    sum(i,0) = 0;
    for (int j=0; j < getNumCols(); j++) {
      sum(i,0) = sum(i,0) + (*this)(i,j);
    }
  }
}

void Matrix::Max(const float scale, Matrix &target) {
  _elemWiseLoop(&_Max, scale, target);
}

float Matrix::Max() const{
  float max = (*this)(0,0); 
  for (int i=0; i < getNumRows(); i++) {
    for (int j=0; j < getNumCols(); j++) {
      if (max < (*this)(i,j))
        max = (*this)(i,j);
    }
  }
  return max;
}

void Matrix::reluGrads(const Matrix &m, Matrix &target) {
  for (int i=0; i < target.getNumRows(); i++) {
    for (int j=0; j < target.getNumCols(); j++) {
      if (m(i,j) >= 0)
        target(i,j) = 1 * (*this)(i,j);
      else
        target(i,j) = 0;
    }
  }
}

void Matrix::sigmoidGrads(const Matrix &m,  Matrix &target) {
  for (int i=0; i < target.getNumRows(); i++) {
    for (int j=0; j < target.getNumCols(); j++) {
        target(i,j) = m(i,j) * (*this)(i,j) * (1 - (*this)(i,j));
    }
  }
}

void Matrix::argmax(float * result) {
  for (int i=0; i < (*this).getNumRows(); i++) {
    float temp = (*this)(i,0);
    result[i] = 0;
    for (int j=1; j < (*this).getNumCols(); j++) {
      if (temp < (*this)(i,j)) {
        temp = (*this)(i,j);
        result[i] = j;
      }
    }
  }
}

Matrix::~Matrix() {
	if(this->_data != NULL) {
		delete[] this-> _data ;
	}
}