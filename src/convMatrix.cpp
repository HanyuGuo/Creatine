#include "../include/convMatrix.hpp"
void convMatrix::_init(float* data, int64 D0, int64 D1, int64 D2, int64 D3) {
  _data = data;
  _D0 = D0;
  _D1 = D1;
  _D2 = D2;
  _D3 = D3;
  _numElements = D0*D1*D2*D3;
}

convMatrix::convMatrix() {
  _init(NULL, 0, 0, 0, 0);
}

convMatrix::convMatrix(int64 D0, int64 D1, int64 D2, int64 D3) {
  _init(NULL, D0, D1, D2, D3);
  this->_data = _numElements > 0 ? new float[this->_numElements] : NULL;
}

convMatrix::convMatrix(float* data, int64 D0, int64 D1, int64 D2, int64 D3) {
  _init(data, D0, D1, D2, D3);
}

void convMatrix::assign(float* data) {
  _data = data;
}

void convMatrix::print_data() {
  for (int64 i = 0; i < _D0; i++) {
    for (int64 j = 0; j < _D1; j++) {
      for (int64 m = 0; m < _D2; m++) {
        for (int64 n = 0; n < _D3; n++) {
          std::cout << getCell(i,j,m,n) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }
}

void convMatrix::convolve(convMatrix &filter, int64 stride, bool samePadding,  convMatrix &target) {
  int64 batch = getDim(0);
  int64 in_height = getDim(1);
  int64 in_width = getDim(2);
  int64 in_channels = getDim(3);
  int64 filter_height = filter.getDim(0);
  int64 filter_width = filter.getDim(1);
  // int64 in_channels = filter.getDim(2);
  int64 out_channels = filter.getDim(3);
  assert(in_channels == filter.getDim(2)); 
  for (int64 b = 0; b < batch; b++) {
  	for (int64 i = 0; i*stride < in_height; i++) {
      for (int64 j = 0; j*stride < in_width; j++) {
        for (int64 k = 0; k < out_channels; k++){
          float pixVal = 0;
          for (int64 c = 0; c < in_channels; c++) {
            int64 startRow = i*stride - filter_height/2;
            int64 startCol = j*stride - filter_width/2;
            for (int64 di = 0; di < filter_height; di++) {
              for (int64 dj = 0; dj < filter_width; dj++) {
                int64 curRow = startRow + di;
                int64 curCol = startCol + dj;
                if (curRow > -1 && curRow < in_height && curCol > -1 && curCol < in_width) {
                  pixVal += (*this)(b, curRow, curCol, c) * filter(di, dj, c, k);
                }
              }
            }
          target(b,i,j,k) = pixVal;
          }
        }
      }
    }
  }
}

void convMatrix::flatten(Matrix &target) {
  for (int64 i = 0; i < _D0; i++) {
    for (int64 j = 0; j < _D1; j++) {
      for (int64 m = 0; m < _D2; m++) {
        for (int64 n = 0; n < _D3; n++) {
          target(i,j*_D2*_D3 + m*_D3 +n) = (*this)(i,j,m,n);
        }
      }
    }
  }
}