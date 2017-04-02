#include <iostream>
#include <vector>
#include <assert.h>
#include <gpmatrix.cuh>
#include <cuda.h>
#include <cublas_v2.h>

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

GpMatrix::GpMatrix(double *devData, int numRows, int numCols, int stride, bool isTrans)
				  :_deviceData(devData),
				  _numRows(numRows),
				  _numCols(numCols),
				  _n_elem(numRows*numCols),
				  _isTrans(isTrans){
				  	stride = stride; // set stride to 1 until clear.
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

/* copy the Gpu matrix to host 
   Note: This assumes that the Matrix type has similar API to GpMatrix.
*/
void GpMatrix::cptoHost(Matrix &mat) const{ 
    assert(checkeqDims(mat));
    if (getNumElements() > 0)
    {
    	 cuBlasStatus_t stat = cublasGetMatrix(getnumRows(),getNumCols(), getnumElements(),
    									  _deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
    	cuBlaserrcheck("Couldn't write to host memory, read failure \n");
    }
    }
   
}

void GpMatrix::cpfromHost(Matrix &mat) const{
  assert(checkeqDims(mat));
    if (getNumElements() > 0)
    {
    	 cuBlasStatus_t stat = cublasSetMatrix(getnumRows(),getNumCols(), getnumElements(),
    									  _deviceData, getLeadingDim(),mat.getData(), mat.getLeadingDim());
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
    	cuBlaserrcheck("Couldn't read from host memory, write failure \n");
    }
    }
}

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


GpMatrix & GpMatrix::slice(int rowStart, int colStart, int rowEnd, int colEnd) const{

}

GpMatrix & GpMatrix::sliceRows(int rowStart, int rowEnd){

}

GpMatrix & GpMatrix::sliceCols(int colStart, int colEnd){

}

void GpMatrix::transpose(GpMatrix &target) {

}
