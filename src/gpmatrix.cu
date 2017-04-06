#include <iostream>
#include <vector>
#include <assert.h>
#include <gpmatrix.cuh>
#include <cuda.h>
#include <cublas_v2.h>
#include "gpmatrix_kernels.h"

#define BLOCK_WIDTH 16


/* RAII baby! */
void GpMatrix::_initGpMatrix(int numRows, int numCols, int stride, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _n_elem= _numRows*_numCols;
    _isTrans = isTrans;
    stride = stride;
    if (_n_elem > 0)
     {
       cuBlasStatus_t stat = cudaMalloc((void**)&_deviceData,_n_elem*sizeof(double));
       if (stat != CUBLAS_STATUS_SUCCESS)
       {
       	 cuBlaserrcheck("Can't allocate memory on GPU\n");
       }
     } 

}

GpMatrix::GpMatrix() {
	_initGpMatrix(0,0,1,0);
}

GpMatrix::GpMatrix(double *devData, int numRows, int numCols,bool isTrans)
				  :_deviceData(devData),
				  _numRows(numRows),
				  _numCols(numCols),
				  _n_elem(numRows*numCols),
				  _isTrans(isTrans){
				  	stride = getLeadingDim(); // set stride to leadingDim until clear.
				  }



GpMatrix::~GpMatrix() {
   if (_n_elem > 0)
   {
   	 cuBlasStatus_t stat = cudaFree((void*)_deviceData);
   	 if (stat != CUBLAS_STATUS_SUCCESS)
   	 {
   	 	cuBlaserrcheck("Can't free memory on GPU. Did you allocate it? \n");
   	 }
   }
}

static int GpMatrix::getDeviceID() {
	int d;
	cudaGetDevice(&d);
	return d;
}
/* copy the Gpu matrix to host 
   Note: This assumes that the Matrix type has similar API to GpMatrix.
*/
// void GpMatrix::cptoHost(Matrix &mat) const{ 
//     assert(checkeqDims(mat));
//     if (getNumElements() > 0)
//     {
//     	 cuBlasStatus_t stat = cublasGetMatrix(getnumRows(),getNumCols(), getnumElements(),
//     									  _deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
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
//     	 cuBlasStatus_t stat = cublasSetMatrix(getnumRows(),getNumCols(), getnumElements(),
//     									  _deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//     	cuBlaserrcheck("Couldn't read from host memory, write failure \n");
//     }
//     }
// }

// bool GpMatrix::checkContiguous(const GpMatrix &mat) {

// }



void GpMatrix::resize(int numRows,int numCols) const{
    if (numRows != _numRows || numRows != _numCols)
    {
    	if (_n_elem > 0)
    	{
    		cuBlasStatus_t stat = cudaFree(_deviceData);
    		if (stat != CUBLAS_STATUS_SUCCESS)
    		{
    		   cuBlaserrcheck("Cannot free the memory..\n");
    		}
    	} 
        if(numRows*numCols > 0) {
			cuBlasStatus_t status = cudaMalloc((void**) _deviceData, numRows*numCols*sizeof(double));
    		if (status != CUBLAS_STATUS_SUCCESS)
    		{
    		   cuBlaserrcheck("Failed to create new resized array \n");
    		}
    	    }else {
    		_deviceData = NULL;
    	}
    	
    _numRows = numRows;
    _numCols = numCols;
    _n_elem = numRows*numCols;
    stride = getLeadingDim();
    }
    

}

GpMatrix & GpMatrix::reshape(int numRows, int numCols)  {
   assert(checkContiguous());
   assert(_n_elem == numRows*numCols); // uh-huh can't reshape into a different matrix
   _numRows = numRows;
   _numCols = numCols;
   stride = getLeadingDim();
   return new *GpMatrix(_deviceData,numRows,numCols,stride, isTrans);

}


void GpMatrix::matCheckBounds(int numRows, int numCols) {
	 assert(numRows>0);
	 assert(numCols>0);
	 assert(numRows==_numRows);
	 assert(numCols==_numCols);
}

void GpMatrix::add(GpMatrix &b, float scaleB, GpMatrix &tgt) {
	assert(a.getNumCols() == b.getNumCols() && a.getNumRows() == b.getNumRows());
    int height = getLeadingDim();
    int width = getFollowingDim();
    if (this->isTrans() == b.isTrans() && tgt.isTrans() == this->isTrans())
    {
    	dim3 blocks(width/ELEM_WISE_THX, height/ELEM_WISE_THY);
    	dim3 threads(ELEM_WISE_THX,ELEM_WISE_THY);
    	MatAdd <<< blocks, threads >>> (getDevData(), b.getDevData(),tgt.getDevData(), height,width,
    		                            getStride(), b.getStride(), tgt.getStride());
    }

	
			
}

void GpMatrix::add(GpMatrix &b, float scale) {
	add(b,scale,*this);
}


void GpMatrix::subtract(GpMatrix &b, float scale, GpMatrix &tgt) {
    add(b,-1,tgt);
}


void GpMatrix::subtract(GpMatrix &b, float scale){
	add(b, -1);
}


/* perform mat mult of the form C = alpha*A*B + beta*C */
void GpMatrix::rightMult(const GpMatrix &b, float scaleAB,GpMatrix &tgt) {
	assert(this->checkContiguous() && b.checkContiguous() && tgt.checkContiguous());
	assert(_numRows == b.getNumCols());

	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	stat = cublasSgemm(handle,this->getTransChar(), b.getTransChar(),_numRows, b.getNumCols(),_numCols,
					   scaleA, _deviceData, getLeadingDim(),b.getDevData(), b.getLeadingDim(),0,
					   tgt.getDevData(), tgt.getLeadingDim());
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cuBlaserrcheck("failed to do matrix multiplication");
	}

}



void GpMatrix::rightMult(const GpMatrix &b, GpMatrix &tgt) {
    rightMult(b, 1, tgt);
}


void GpMatrix::rightMult(const GpMatrix &b, float scale) {
	rightMult(b,1,*this);
}


/* Matrices are in column major order. */
void GpMatrix::addProduct(const GpMatrix &a, const GpMatrix &b, float scaleAB, float scaleC){
	assert(a.checkContiguous() && b.checkContiguous());
	assert(a.getNumCols() == b.getNumRows()); // check if a & b can be multiplied.
	if (scaleC == 0)
	{
		a.rightMult(b,scaleAB,*this);
		return;
	}
    cublasHandle_t handle;
    cuBlasStatus_t stat;
    cublasCreate(&handle);
    stat = cublasSgemm(handle, a.getTransChar(),b.getTransChar(),a.getNumRows(),b.getNumCols(),_numCols,
    	               scaleAB,
    				   a.getDevData(), a.getLeadingDim(),
    				   b.getDevData(), b.getLeadingDim(),
    				   scaleC,
    				   _devData, _numCols);
if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cuBlaserrcheck("failed to do matrix multiplication");
	}
    
}