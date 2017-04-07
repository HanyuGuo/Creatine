#include <iostream>
#include <vector>
#include <assert.h>

#include <cuda.h>
#include <cublas_v2.h>
//#include "../include/gpmatrix_kernels.cuh"
#include "../include/gpmatrix.cuh"

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
       cudaError_t stat = cudaMalloc((void**)&_deviceData,_n_elem*sizeof(double));
       if (stat != cudaSuccess)
       {
       	 cuBlaserrcheck("Can't allocate memory on GPU\n");
       }
     } 

}

GpMatrix::GpMatrix() {
	_initGpMatrix(0,0,1,0);
}

GpMatrix::GpMatrix(float *devData, int numRows, int numCols,bool isTrans):
                  _deviceData(devData),
				  _numRows(numRows),
				  _numCols(numCols),
				  _n_elem(numRows*numCols),
				  _isTrans(isTrans){
				  	stride = getLeadingDim(); // set stride to leadingDim until clear.
	               }



GpMatrix::~GpMatrix() {
   if (_n_elem > 0)
   {
   	 cudaError_t stat = cudaFree((void*)_deviceData);
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
//     	 cublasStatus_t stat = cublasSetMatrix(getnumRows(),getnumCols(), getnumElements(),
//     									  _deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
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
    		cudaError_t stat = cudaFree(_deviceData);
    		if (stat != cudaSuccess)
    		{
    		   cuBlaserrcheck("Cannot free the memory..\n");
    		}
    	} 
        if(numRows*numCols > 0) {
			cudaError_t status = cudaMalloc((void**) _deviceData, numRows*numCols*sizeof(double));
    		if (status != cudaSuccess)
    		{
    		   cuBlaserrcheck("Failed to create new resized array \n");
    		}
    	    } else {
    		_deviceData = NULL;
    	}
    	
    _numRows = numRows;
    _numCols = numCols;
    _n_elem = numRows*numCols;
    stride = getLeadingDim();
    }
    

}

// GpMatrix & GpMatrix::reshape(int numRows, int numCols)  {
//    assert(checkContiguous());
//    assert(_n_elem == numRows*numCols); // uh-huh can't reshape into a different matrix
//    _numRows = numRows;
//    _numCols = numCols;
//    stride = getLeadingDim();
//    return new *GpMatrix(_deviceData,numRows,numCols,stride, isTrans);

// }


void GpMatrix::matCheckBounds(int numRows, int numCols) const {
	 assert(numRows>0);
	 assert(numCols>0);
	 assert(numRows==_numRows);
	 assert(numCols==_numCols);
}

// void GpMatrix::add(GpMatrix &b, float scaleB, GpMatrix &tgt) {
// 	assert(a.getnumCols() == b.getnumCols() && a.getnumRows() == b.getnumRows());
//     int height = getLeadingDim();
//     int width = getFollowingDim();
//     if (this->checkTrans() == b.checkTrans() && tgt.isTrans() == this->checkTrans())
//     {
//     	dim3 blocks(width/ELEM_WISE_THX, height/ELEM_WISE_THY);
//     	dim3 threads(ELEM_WISE_THX,ELEM_WISE_THY);
//     	MatAdd <<< blocks, threads >>> (getDevData(), b.getDevData(),tgt.getDevData(), height,width,
//     		                            getStride(), b.getStride(), tgt.getStride());
//     }

	
			
// }

// void GpMatrix::add(GpMatrix &b, float scale) {
// 	add(b,scale,*this);
// }


// void GpMatrix::subtract(GpMatrix &b, float scale, GpMatrix &tgt) {
//     add(b,-1,tgt);
// }


// void GpMatrix::subtract(GpMatrix &b, float scale){
// 	add(b, -1);
// }


/* perform mat mult of the form C = alpha*A*B + beta*C */
void GpMatrix::RightMult(const GpMatrix &b,float scaleAB, GpMatrix &tgt) {
	assert(this->checkContiguous() && b.checkContiguous() && tgt.checkContiguous());
	assert(_numRows == b.getnumCols());

	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	stat = cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,_numRows, b.getnumCols(),_numCols,
					   &scaleAB, _deviceData, getLeadingDim(),b.getDevData(), b.getLeadingDim(),0,
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
    				   _deviceData, _numCols);
if (stat != CUBLAS_STATUS_SUCCESS)
	{
		cuBlaserrcheck("failed to do matrix multiplication");
	}

}


void GpMatrix::printMat(int numRows, int numCols){
	for (int i = 0; i < numRows; ++i)
	{
		for (int j = 0; j < numCols; ++j)
		{
			std::cout<<"i: "<<i << " j: "<< j<< " Mat: " << _deviceData[j*numRows+i] <<"\n";
		}
	}
}