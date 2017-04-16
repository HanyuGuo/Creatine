#include <iostream>
#include <assert.h>
#include "../../include/convlayer.hpp"
#include "../../include/cudaMatrix.hpp"

ConvLayer::ConvLayer(float *data, int numrows, int numcols, int padding, int stride,Activation a){
  ldata = new cudaMatrix(data,numrows,numcols);
  height = numrows;
  width = numrows;
  padding = pad;
  stride = stride;

}
