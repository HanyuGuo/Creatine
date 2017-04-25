#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "../include/cudaMatrix.cuh"


#define GPU


void cudaMatrix::_init(float *data, int numrows, int numcols){
  devData = NULL;
  numRows = numrows;
  numCols = numcols;
  numElems = numRows * numCols;
  cudaError_t err;
  setbestcudaDevice();
  cudaSetDevice(gpuid);
  if (data != NULL) {
    // std::cout << "got data!\n";
    if (numElems > 0) {
      cudaMalloc((void**)&devData, numElems*sizeof(float));
      cudaMemcpy(devData,data,numElems*sizeof(float),cudaMemcpyHostToDevice);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
      }
    }
  }
  else {
    // std::cout << "empty data!\n";
    if (numElems > 0) {
      cudaMalloc((void**)&devData, numElems*sizeof(float));
      err = cudaGetLastError();
      if (err != cudaSuccess) {
       std::cout << "Couldn't allocate memory\n";
      }
    }
  }
}

cudaMatrix::cudaMatrix(int numrows, int numcols){
  _init(NULL,numrows,numcols);
}


cudaMatrix::cudaMatrix(float *data, int numrows, int numcols){
   _init(data,numrows,numcols);
}


cudaMatrix::~cudaMatrix(){
  cudaError_t err;
  if (numElems > 0 ) {
      cudaFree(devData);
      err = cudaGetLastError();

      if (err != cudaSuccess) {
        std::cout << "Can't free memory\n";
      }
  }
}

 void cudaMatrix::setbestcudaDevice() {
  int num_dev;
  cudaGetDeviceCount(&num_dev);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  long max = props.totalGlobalMem;
  gpuid = 0;
  // std :: cout << "mem " << max << "\n";
  std::vector<cudaDeviceProp> dev_props;
   for (int i = 1; i < num_dev; i++) {
     cudaGetDeviceProperties(&props, i);
     // std :: cout << "mem " << props.totalGlobalMem << "\n";
     if (max < props.totalGlobalMem) {
       max = props.totalGlobalMem;
       gpuid = i;
     }
  }
}

void cudaMatrix::setDeviceData(float *data, int elems) {
  cudaError_t err;

  if (elems != numElems) {
    std::cout<< "The size of data must be same! Aborting..\n";
    exit(1);
  }

  cudaMemcpy(devData,data,elems*sizeof(float),cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't write to device data." << '\n';
  }
}


void cudaMatrix::getDeviceData(float *hdata) {
  cudaError_t err;
  // if (hdata == NULL) {
  //    *hdata = new float[numElems];
  // }
  cudaMemcpy(hdata,devData, numElems*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Can't read device data." << '\n';
  }
}



 void cudaMatrix::cudaAdd(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x,grid_dim_y, 1);
    dim3 block(block_dim_x, block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    // std::cout<<"Launching kernel now...\n";
    MatAddKernel<<<grid, block>>>(adata,bdata,resdata,numRows, numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }
  } else {
    printf("Matrix dims must be same, aborting..\n");
    exit(1);
  }

}


void cudaMatrix::cudaWeightedAdd(const cudaMatrix &b, cudaMatrix &c, float scale) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x,grid_dim_y, 1);
    dim3 block(block_dim_x, block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    // std::cout << "Launching Weighted Add..." << '\n';
    WeightedAddKernel<<< grid, block >>>(adata,bdata,resdata,scale,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }
  } else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}



void cudaMatrix::cudaElemWiseMult(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x, grid_dim_y,1);
    dim3 block(block_dim_x,block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout<<"Doing Elt wise mat mul..."<<"\n";
    EltWiseMatMul<<< grid, block >>> (adata, bdata, resdata,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }

  }else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}


void cudaMatrix::cudaElemWiseDivide(const cudaMatrix &b, cudaMatrix &c) {
  cudaError_t err;
  if (numRows == b.getNumRows() && numCols == b.getNumCols()) {
    int block_dim_x = 32;
    int block_dim_y = 32;
    int grid_dim_x = (numRows*numCols)/block_dim_x;
    int grid_dim_y = (numRows*numCols)/block_dim_y;
    dim3 grid(grid_dim_x, grid_dim_y,1);
    dim3 block(block_dim_x,block_dim_y);
    float *adata = devData;
    float *bdata = b.getDevData();
    float *resdata = c.getDevData();
    std::cout<<"Doing Elt wise mat divide..."<<"\n";
    EltWiseMatDivide<<< grid, block >>> (adata, bdata, resdata,numRows,numCols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error launching kernel\n");
    }

  }else {
    printf("Matrix dims must be the same, aborting...\n");
    exit(1);
  }
}


void cudaMatrix::powgpu(int scale){
  cudaError_t err;
  int block_x = 512; // for max threads per block
  int grid_x = (numElems-1)/block_x;
  // int grid_y = (numElems-1)/block_x;
  dim3 grid(grid_x,1,1);
  dim3 block(block_x,1,1);
  powgpu_kernel <<<grid, block >>>(devData,numElems,scale);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Couldn't launch power gpu kernel..\n");
  }

}



void cudaMatrix::expgpu(cudaMatrix &tgt) {
  cudaError_t err;
  int block_dim_x = 32;
  int block_dim_y = 32;
  int grid_dim_x = numRows/block_dim_x + 1;
  int grid_dim_y = numCols/block_dim_y + 1;
  // std:: cout << numRows << " " << numCols << "\n";
  dim3 grid(grid_dim_x, grid_dim_y,1);
  dim3 block(block_dim_x,block_dim_y);
  expgpu_kernel <<<grid, block >>>(devData, numElems, tgt.getDevData());
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Couldn't launch exp gpu kernel..\n");
  }
}


void cudaMatrix::axpy_ongpu(const cudaMatrix &b, float scaleA, int ldx, int ldy,cudaMatrix &tgt){
  cudaError_t err;
  int block_x = 512;
  int grid_x = (numElems-1)/block_x;
  dim3 grid(grid_x,1,1);
  dim3 block(block_x,1,1);
  printf("Launching axpy kernel...\n");
  axpy_kernel <<< grid, block >>> (devData,b.getDevData(),scaleA,ldx,ldy,numElems,tgt.getDevData());
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Couldn't launch axpy gpu kernel..\n");
  }
}

// float* cudaMatrix::reshape_data(float *data, int numRows, int numCols) {
//     float *rdata;
//
//     cudaMalloc((void**)&rdata, numRows*numCols*sizeof(float));
//     for (int i = 0; i < numRows; i++) {
//       for (int j = 0; j < numCols; j++) {
//          rdata[ci(i,j,numCols)] = data[i*numRows+j];
//       }
//     }
//     return rdata;
// }

void cudaMatrix::gemm_ongpu(bool tA, bool tB, const cudaMatrix &b, float scaleA, float scaleB, cudaMatrix &tgt){
  cudaError_t err;
  int m = b.getNumCols();
  int n = numRows;
  int k = numCols;
  cublasHandle_t handle;
  cublasCreate(&handle);
  if ((tgt.getNumCols() != b.getNumCols()) || (tgt.getNumRows()!=numRows)) {
     std::cout << "Matrix dimensions are not same .. aborting \n";
     exit(1);
  }

  cublasSgemm(handle,(tA?CUBLAS_OP_T:CUBLAS_OP_N),(tB?CUBLAS_OP_T:CUBLAS_OP_N),
              m,n,k,&scaleA,b.getDevData(),b.getNumCols(),devData,numCols,&scaleB,tgt.getDevData(),b.getNumCols());
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cannot do Sgemm..\n");
  }

}
void cudaMatrix::cudaAddv(const cudaMatrix &b, float scale, cudaMatrix &c) {
    cudaError_t err;
    // int block_dim_x = 4;
    // int block_dim_y = 4;
    // int grid_dim_x = (numRows*numCols)/block_dim_x;
    // int grid_dim_y = (numRows*numCols)/block_dim_y;

    // dim3 grid(grid_dim_x, grid_dim_y,1);
    // dim3 block(block_dim_x,block_dim_y);
    // add_mat_vec_kernel <<<grid, block >>> (devData,b.getDevData(),numRows,numCols,scale,c.getDevData());
    int block_dim_x = 512;
    int grid_dim_x = (numRows*numCols)/block_dim_x + 1;

    dim3 grid(grid_dim_x, 1,1);
    dim3 block(block_dim_x,1);
    add_mat_vec_kernel <<<grid, block >>> (devData,b.getDevData(),numRows,numCols,c.getDevData());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("can't add matrices and vectors.\n");
    }



}


void cudaMatrix::calc_activation_gpu(Activation a, cudaMatrix &tgt) {
   activations_on_gpu(devData,numElems,a,tgt.getDevData());
}


void cudaMatrix::cudaDivideByVector(const cudaMatrix &b, cudaMatrix &tgt){
     cudaError_t err;
     int block_x = 32;
     int block_y = 32;
     int grid_x = (numElems)/block_x;
     dim3 grid(grid_x,1,1);
     dim3 block(block_x,block_y);
     DivideByScalar <<< grid, block >>> (devData,b.getDevData(),numRows, numRows, tgt.getDevData());
     err = cudaGetLastError();
     printf("err %d",err);
     if (err != cudaSuccess) {
       printf("can't divide by scalar.\n");
     }
}


/* b must be a unit vector for this to work! */
void cudaMatrix::softmax_gpu(cudaMatrix &tgt){
  // cudaError_t err1,err2,err3,err4,err5;
  float * temp_max;
  cudaMalloc((void**)&temp_max, numRows*sizeof(float));
  float * temp_subtract;
  cudaMalloc((void**)&temp_subtract, numRows*numCols*sizeof(float));
  float * temp_exp;
  cudaMalloc((void**)&temp_exp, numRows*numCols*sizeof(float));
  // std::cout << "calculating max....";
  int block,grid;
  this->getkernelConfig(true,&block, &grid);
  calc_max <<<grid,block>>>(numRows,numCols,devData,temp_max);
  // err1 = cudaGetLastError();
  // std::cout<< "err1 "<< err1;
  this->getkernelConfig(false, &block, &grid);
  subtract_max <<<grid,block>>>(numRows,numCols,devData,temp_max, temp_subtract);
  // err2 = cudaGetLastError();
  // std::cout<< "err2 "<< err2;
  this->getkernelConfig(false, &block, &grid);
  expgpu_kernel<<<grid,block>>>(temp_subtract,numRows*numCols, temp_exp);
  // err3 = cudaGetLastError();
  // std::cout<<"err3 "<<err3;
  this->getkernelConfig(true, &block, &grid);
  calc_sum_row<<<grid,block>>>(numRows,numCols,temp_exp, temp_max);
  // err4 = cudaGetLastError();
  // std::cout<< "err4 "<< err4;
  this->getkernelConfig(false, &block, &grid);
  div_row<<<grid,block>>>(numRows,numCols,temp_exp,temp_max,tgt.getDevData());
  // err5 = cudaGetLastError();
  // std::cout<< "err5 "<< err5; 
  // cudaFree(scale);
  cudaFree(temp_max);
  cudaFree(temp_subtract);
  cudaFree(temp_exp);

}


void cudaMatrix::argmax_gpu(int* result) {
  int block,grid;
  int * temp_result;
  cudaMalloc((void**)&temp_result, numRows*sizeof(int));
  this->getkernelConfig(true, &block, &grid);
  argmax <<<grid,block>>>(numRows,numCols,devData, temp_result);
  cudaMemcpy(result,temp_result, numRows*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(temp_result);

}