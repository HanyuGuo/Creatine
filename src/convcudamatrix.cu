#include <iostream>
#include "../include/convcudamatrix.cuh"


void cudaConvMatrix::_init(float *data, int D0, int D1, int D2, int D3){
	devData = NULL;
  _D0 = D0;
  _D1 = D1;
  _D2 = D2;
  _D3 = D3;
  _numElements = D0*D1*D2*D3;
  cudaError_t err;
  ret_data = new float[_numElements];
  if (data != NULL) {
    // std::cout << "got data!\n";
    if (_numElements > 0) {
      cudaMalloc((void**)&devData, _numElements*sizeof(float));
      cudaMemcpy(devData,data,_numElements*sizeof(float),cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
      }
    }
  }
  else {
    // std::cout << "empty data!\n";
    if (_numElements > 0) {
      cudaMalloc((void**)&devData, _numElements*sizeof(float));
      err = cudaGetLastError();
      if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
      }
    }
  }
}


cudaConvMatrix::cudaConvMatrix(){
	_init(NULL,0,0,0,0);
}

cudaConvMatrix::cudaConvMatrix(int D0, int D1, int D2, int D3){
	_init(NULL, D0, D1, D2, D3);
}

cudaConvMatrix::cudaConvMatrix(float* data, int D0, int D1, int D2, int D3){
	_init(data, D0, D1, D2, D3);
}


cudaConvMatrix::~cudaConvMatrix()
{
  cudaError_t err;
  if (_numElements > 0 ) {
      cudaFree(devData);
      err = cudaGetLastError();

      if (err != cudaSuccess) {
        std::cout << "Can't free memory\n";
      }
  }
  delete [] ret_data;
}


// void cudaConvMatrix::fwd_pass_convolution_gpu(cudaConvMatrix &weights, cudaConvMatrix &tgt){
//   cudaError_t err;
// 	int col_height = ((_height-_kern_sz/2)/_stride)*((_width-_kern_sz/2)/_stride);
// 	int col_width = _kern_sz*_kern_sz*_ch_in;
// 	float *col;
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     int m = getnDim(2);
//     int n = col_height*col_width;
//     int k = _kern_sz*_kern_sz*_ch_in;

// 	cudaMalloc((void**)&col,col_width*col_height*sizeof(float));
// 	for (int i = 0; i < _bs; ++i)
// 	{  
//     im2col_gpu(devData+i*_height*_width*_ch_in,_ch_in, _height,_width,_kern_sz,_stride,_pad,col_height,col_width,col);
//     cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,m,n,k,&scaleA,weights.getDevData(),k,col,n,&scaleB,tgt.getDevData()+i*m*n,n); 
//      err = cudaGetLastError();
//     if (err != cudaSuccess) {
//       printf("Cannot do Sgemm..\n");
//     }      
//     }
//   cudaFree(col);
// }


void cudaConvMatrix::convolve(cudaConvMatrix &weights, bool samePadding, int stride, cudaConvMatrix &tgt){
  cudaError_t err;
  int batch = getDim(0);
  int in_height = getDim(1);
  int in_width = getDim(2);
  int in_channels = getDim(3);
  int filter_height = weights.getDim(0);
  int filter_width = weights.getDim(1);
  // int in_channels = filter.getDim(2);
  int out_channels = weights.getDim(3);
  assert(in_channels == weights.getDim(2)); 
  int out_height = tgt.getDim(1);
  int out_width = tgt.getDim(2);
  // std::cout << batch << " " << in_height << " " << in_width << " " << in_channels << "\n"; 
  // std::cout << filter_height << " " << filter_height << " " << in_channels << " " <<  out_channels <<"\n"; 

  // int col_height = ((in_height-filter_height/2)/stride)*((in_width-filter_width/2)/stride);
  // int col_width = filter_height*filter_width*in_channels;
  // std::cout << "col_height:" << col_height << " col_width: " <<  col_width << "\n";
  int m = out_channels; 
  int k = filter_height * filter_width * in_channels;
  int n = out_height * out_width;
  float* temp;
  cudaMalloc((void**)&temp, out_channels*out_height*out_width*sizeof(float));
  cublasHandle_t handle;
  float scaleA = 1.f;
  float scaleB = 1.f;
  for(int i = 0; i < batch; i++) { 
    im2col_ongpu(devData + i*in_height*in_width*in_channels,
                 in_channels, in_height, in_width, filter_height, stride,
                 filter_height/2, temp);
    // float* peek = new float[5];
    // cudaMemcpy(peek,devData,5*sizeof(float),cudaMemcpyDeviceToHost);
    // std::cout << peek[0] << peek[1] << peek[2] << peek[3] << peek[4];
    // cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,m,n,k,&scaleA,weights.getDevData(),k,temp,n,&scaleB,tgt.getDevData()+i*m*n,n); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Cannot do Sgemm..\n");
    }   
  }
  cudaFree(temp);
  

}
