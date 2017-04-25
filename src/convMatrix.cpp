#include "../include/convMatrix.hpp"
void convMatrix::_init(float* data, int D0, int D1, int D2, int D3) {
  _data = data;
  _D0 = D0;
  _D1 = D1;
  _D2 = D2;
  _D3 = D3;
  _numElements = D0*D1*D2*D3;
}

void convMatrix::_init_gpu(float *data, int D0,int D1, int D2, int D3){
  cudaError_t err;
  _data = NULL;
  _D0 = D0;
  _D1 = D1;
  _D2 = D2;
  _D3 = D3;
  _numElements = D0*D1*D2*D3;
   if (data != NULL) {
    std::cout << "got data!\n";
    if (_numElements > 0) {
      cudaMalloc((void**)&_data, _numElements*sizeof(float));
      cudaMemcpy(_data,data,_numElements*sizeof(float),cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
      }
    }
  }
}


convMatrix::convMatrix() {
  _init(NULL, 0, 0, 0, 0);
}

convMatrix::convMatrix(int D0, int D1, int D2, int D3) {
  _init(NULL, D0, D1, D2, D3);
  this->_data = _numElements > 0 ? new float[this->_numElements] : NULL;
}

convMatrix::convMatrix(float* data, int D0, int D1, int D2, int D3, bool isGPU) {
  if (isGPU)
  {
    _init_gpu(data,D0,D1,D2,D3);
  } else {
    _init(data, D0, D1, D2, D3);
  }
  
    
  
  
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
  	for (int i = 0; i*stride < in_height; i++) {
      for (int j = 0; j*stride < in_width; j++) {
        for (int k = 0; k < out_channels; k++){
          float pixVal = 0;
          for (int c = 0; c < in_channels; c++) {
            int startRow = i*stride - filter_height/2;
            int startCol = j*stride - filter_width/2;
            for (int di = 0; di < filter_height; di++) {
              for (int dj = 0; dj < filter_width; dj++) {
                int curRow = startRow + di;
                int curCol = startCol + dj;
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

// convMatrix & convMatrix::sliceMatrix(int height, int width, int startRow, int startCol, int endRow, int endCol, int channels, bool isGPU) const {
//    if (isGPU)
//    {
     
//    } else {
//       return *new convMatrix(this->_data+)
//    }


// }
void convMatrix::fwdpassconvgpu(cudaMatrix &weights, cudaMatrix &col, int stride, int padding, int kern_sz, int col_height, int col_width, cudaMatrix &tgt){
    int bs = this->getDim(0);
    int height = this->getDim(1);
    int width = this->getDim(2);
    int channels = this->getDim(3);


    for (int i = 0; i < this->getDim(0); ++i) {
        im2col_gpu(_data,channels,height,width,kern_sz,stride,pad,col_height,col_width,col.getDevData()); // convert an ip heightxwidthxchannel 3d matrix to a 2d one.
        col.gemm_ongpu(false,false,weights,1,0,tgt);
    }
}
