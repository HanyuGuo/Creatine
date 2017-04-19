#include <iostream>
#include <vector>
#include <assert.h>

#include <cuda.h>
#include <cublas_v2.h>
#include "../include/gpmatrix_kernels.cuh"
#include "../include/gpmatrix.cuh"

#define BLOCK_WIDTH 16


void GpMatrix::_initGpMatrix(float *devData,int numRows, int numCols, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _n_elem= _numRows*_numCols;
    _isTrans = isTrans;
    deviceData = devData;
    if (_n_elem > 0)
    {
    	cudaMalloc((void**)&deviceData,_n_elem*sizeof(float));
    	//cuBlaserrcheck("failed to init matrix!!\n");
    }

    stride = getLeadingDim();

}


/*
Init the matrix and set the parameters.
 */
GpMatrix::GpMatrix(int numRows, int numCols,bool isTrans){
           _initGpMatrix(NULL,numRows,numCols,true);
           if (numRows * numCols > 0)
           {
           	  cudaError_t stat = cudaMalloc((void**)&deviceData, numRows*numCols*sizeof(float));
           	  if (stat != cudaSuccess)
           	  {
           	  	 std::cout<<"Can't allocate matrix!\n";
           	  }
           }

}

GpMatrix::GpMatrix() {
	_initGpMatrix(NULL,0,0,false);
}


GpMatrix::GpMatrix(float *devData, int numRows,int numCols, bool isTrans){
	_initGpMatrix(devData,numRows,numCols, isTrans);
}



GpMatrix::~GpMatrix() {
   if (_n_elem > 0)
   {
   	 cudaError_t stat = cudaFree((void*)deviceData);
   	 if (stat != cudaSuccess)
   	 {
   	 	cuBlaserrcheck("Can't free memory on GPU. Did you allocate it? \n");
   	 }
   }
}

// static int GpMatrix::getDeviceID() {
// 	int d;
// 	cudaGetDevice(&d);
// 	return d;
// }
/* copy the Gpu matrix to host
   Note: This assumes that the Matrix type has similar API to GpMatrix.
*/
// void GpMatrix::cptoHost(Matrix &mat) const{
//     assert(checkeqDims(mat));
//     if (getNumElements() > 0)
//     {
//     	 cublasStatus_t stat = cublasGetMatrix(getnumRows(),getnumCols(), getnumElements(),
//     									  deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//     	cuBlaserrcheck("Couldn't write to host memory, read failure \n");
//     }
//     }

// }

// void GpMatrix::cpfromHost(Matrix &mat) const{
//   assert(checkeqDims(mat));
//     if (getNumElements() > 0)
//     {
//     	 cublasStatus_t stat = cublasSetMatrix(getnumRows(),getnumCols(), getnumElements(),
//     									  deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//     	cuBlaserrcheck("Couldn't read from host memory, write failure \n");
//     }
//     }
// }

// bool GpMatrix::checkContiguous(const GpMatrix &mat) {

// }



void GpMatrix::resize(int numRows,int numCols) {
    if (numRows != _numRows || numRows != _numCols)
    {
    	if (_n_elem > 0)
    	{
    		cudaError_t stat = cudaFree(deviceData);
    		if (stat != cudaSuccess)
    		{
    		   cuBlaserrcheck("Cannot free the memory..\n");
    		}
    	}
        if(numRows*numCols > 0) {
			cudaError_t status = cudaMalloc((void**) deviceData, numRows*numCols*sizeof(float));
    		if (status != cudaSuccess)
    		{
    		   cuBlaserrcheck("Failed to create new resized array \n");
    		}
    	    } else {
    		deviceData = NULL;
    	}

    _numRows = numRows;
    _numCols = numCols;
    _n_elem = numRows*numCols;
    stride = getLeadingDim();
    }


}



void GpMatrix::matCheckBounds(int numRows, int numCols) const {
	 assert(numRows>0);
	 assert(numCols>0);
	 assert(numRows==_numRows);
	 assert(numCols==_numCols);
}

void GpMatrix::add(const GpMatrix &b, float scaleB, GpMatrix &tgt) {
    cudaError err;
	assert(getnumCols() == b.getnumCols() && getnumRows() == b.getnumRows());
    int height = getLeadingDim();
    int width = getFollowingDim();
    if(getDevData() == NULL) {
      std::cout<<"No data available on GPU\n";
    }
    if (checkTrans() == b.checkTrans() && tgt.checkTrans() == checkTrans())
    {   std::cout<<"Launching kernel...\n";
    	dim3 blocks(width/ELEM_WISE_THX, height/ELEM_WISE_THY);
    	dim3 threads(ELEM_WISE_THX,ELEM_WISE_THY);
    	MatAdd <<< blocks, threads >>> (getDevData(), b.getDevData(),tgt.getDevData(), height,width);
    	err = cudaGetLastError();
    	if (err != cudaSuccess)
    	{
    	   std::cout<<"Error launching kernel...\n";
    	}
    }



}

void GpMatrix::add(const GpMatrix &b, float scale) {
	add(b, scale, *this);
}

void GpMatrix::add(const GpMatrix &b){
	add(b,1,*this);
}

// void GpMatrix::subtract(GpMatrix &b, float scale, GpMatrix &tgt) {
//     add(b,-1,tgt);
// }


// void GpMatrix::subtract(GpMatrix &b, float scale){
// 	add(b, -1);
// }


/* perform mat mult of the form C = alpha*A*B + beta*C */
void GpMatrix::RightMult(const GpMatrix &b,float scaleAB, GpMatrix &tgt) {
	assert(checkContiguous() && b.checkContiguous() && tgt.checkContiguous());
	assert(_numRows == b.getnumCols());

	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,_numRows, b.getnumCols(),_numCols,
					   &scaleAB, deviceData, getLeadingDim(),b.getDevData(), b.getLeadingDim(),0,
					   tgt.getDevData(), tgt.getLeadingDim());
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cuBlaserrcheck("failed to do matrix multiplication");
	}

}



void GpMatrix::RightMult(const GpMatrix &b, GpMatrix &tgt) {
    RightMult(b, 1, tgt);
}


void GpMatrix::RightMult(const GpMatrix &b, float scale) {
	RightMult(b,1,*this);
}


/* Matrices are in column major order. */
void GpMatrix::addProduct(const GpMatrix &a, const GpMatrix &b, float scaleAB, float scaleC){
	assert(a.checkContiguous() && b.checkContiguous());
	assert(a.getnumCols() == b.getnumRows()); // check if a & b can be multiplied.
	// if (scaleC == 0)
	// {
	// 	a.RightMult(b,scaleAB,*this);
	// 	return;
	// }
    cublasHandle_t handle;
    cublasStatus_t stat;
    cublasCreate(&handle);
    stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,a.getnumRows(),b.getnumCols(),_numCols,
    	               &scaleAB,
    				   a.getDevData(), a.getLeadingDim(),
    				   b.getDevData(), b.getLeadingDim(),
    				   &scaleC,
    				   deviceData, _numCols);
if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cuBlaserrcheck("failed to do matrix multiplication");
	}
    cublasDestroy(handle);
}


void GpMatrix::printMat(int numRows, int numCols){
	for (int i = 0; i < numRows; ++i)
	{
		for (int j = 0; j < numCols; ++j)
		{
			std::cout<<"i: "<<i << " j: "<< j<< " Mat: " << deviceData[j*numRows+i] <<"\n";
		}
	}
}
