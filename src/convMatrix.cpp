#include "../include/convMatrix.hpp"
void convMatrix::_init(float* data, int D0, int D1, int D2, int D3) {
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

convMatrix::convMatrix(int D0, int D1, int D2, int D3) {
  _init(NULL, D0, D1, D2, D3);
  this->_data = _numElements > 0 ? new float[this->_numElements] : NULL;
}

convMatrix::convMatrix(float* data, int D0, int D1, int D2, int D3) {
  _init(data, D0, D1, D2, D3);
}

void convMatrix::assign(float* data) {
  _data = data;
}

void convMatrix::print_data() {
  for (int i = 0; i < _D0; i++) {
    for (int j = 0; j < _D1; j++) {
      for (int m = 0; m < _D2; m++) {
        for (int n = 0; n < _D3; n++) {
          std::cout << getCell(i,j,m,n) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }
}

void convMatrix::convolve(convMatrix &filter, int stride, bool samePadding,  convMatrix &target) {
  int batch = getDim(0);
  int in_height = getDim(1);
  int in_width = getDim(2);
  int in_channels = getDim(3);
  int filter_height = filter.getDim(0);
  int filter_width = filter.getDim(1);
  // int in_channels = filter.getDim(2);
  int out_channels = filter.getDim(3);
  assert(in_channels == filter.getDim(2)); 
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < out_channels; k++) {
      float pixVal = 0;
    	for (int i = 0; i*stride < in_height; i++) {
        for (int j = 0; j*stride < in_width; j++) {
          for (int c = 0; c < in_channels; c++) {
            int startRow = i*stride;// - filter_height/2;
            int startCol = j*stride;// - filter_width/2;
            for (int di = 0; di < filter_height; di++) {
              for (int dj = 0; dj < filter_width; dj++) {
                int curRow = startRow + di;
                int curCol = startCol + dj;
                if (curRow > -1 && curRow < in_height && curCol > -1 && curCol < in_width) {
                  pixVal += (*this)(b, curRow, curCol, c) * filter(di, dj, c, k);
                }
              }
            }
          }
          target(b,i,j,k) = pixVal;        
        }
      }
    }
  }
}


// void convMatrix::convolve(convMatrix &filter, int stride, bool samePadding,  convMatrix &target) {
//   int batch = getDim(0);
//   int in_height = getDim(1);
//   int in_width = getDim(2);
//   int in_channels = getDim(3);
//   int filter_height = filter.getDim(0);
//   int filter_width = filter.getDim(1);
//   // int in_channels = filter.getDim(2);
//   int out_channels = filter.getDim(3);
//   assert(in_channels == filter.getDim(2)); 
//   int pad_h = filter_height/2;
//   int pad_w = filter_w/2;
//   const int output_h = (height + 2 * pad_h -
//     (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
//   const int output_w = (width + 2 * pad_w -
//     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
//   const int channel_size = height * width;
//   for (int channel = channels; channel--; data_im += channel_size) {
//     for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
//       for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
//         int input_row = -pad_h + kernel_row * dilation_h;
//         for (int output_rows = output_h; output_rows; output_rows--) {
//           if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
//             for (int output_cols = output_w; output_cols; output_cols--) {
//               *(data_col++) = 0;
//             }
//           } else {
//             int input_col = -pad_w + kernel_col * dilation_w;
//             for (int output_col = output_w; output_col; output_col--) {
//               if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
//                 *(data_col++) = data_im[input_row * width + input_col];
//               } else {
//                 *(data_col++) = 0;
//               }
//               input_col += stride_w;
//             }
//           }
//           input_row += stride_h;
//         }
//       }
//     }
//   }
// }

void convMatrix::flatten(Matrix &target) {
  for (int i = 0; i < _D0; i++) {
    for (int j = 0; j < _D1; j++) {
      for (int m = 0; m < _D2; m++) {
        for (int n = 0; n < _D3; n++) {
          target(i,j*_D2*_D3 + m*_D3 +n) = (*this)(i,j,m,n);
        }
      }
    }
  }
}


void convMatrix::max(const int scale, convMatrix &target) {
  for (int i = 0; i < _D0; i++) {
    for (int j = 0; j < _D1; j++) {
      for (int m = 0; m < _D2; m++) {
        for (int n = 0; n < _D3; n++) {
          target(i,j,m,n) = _max((*this)(i,j,m,n), scale);
        }
      }
    }
  }
}

convMatrix::~convMatrix() {
  if(this->_data != NULL) {
    // delete[] this-> _data ;
  }
} 