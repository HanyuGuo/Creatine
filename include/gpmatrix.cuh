#ifndef GPMATRIX_HPP_
#define GPMATRIX_HPP_ 
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cublas_v2.h>
//#include <helper_cuda.h>
#include <cmath>


// #define cudaCheckError() __cudaCheckError(__FILE__,__LINE__)
// #define cudaSafeCall(err) __cudaSafeCall(err,__FILE__,__LINE__)

// inline void __cudaSafeCall(cudaError err,const char*file,const int line) {
//    if(cudaSuccess != err) {
//     fprintf(stderr,"cudaSafeCall() failed at %s:%i: %s\n",file,line,cudaGetErrorString(err));
//     exit(-1);
//    }
//  }

// inline void __cudaCheckError(const char*file,const int line) {
//    cudaError err = cudaGetLastError();
//    if(cudaSuccess != err) {
//       printf("cudaCheckError() failed at %s:%i :%s \n", file,line,cudaGetErrorString(err));
//       exit(-1);
//     }
//   }


/*
 GpMatrix: Class encapsulating the behavior of the 
 CUDA matrix datatype. 

 Adds functions to use with the GpMatrix object.

 Note: In cuBLAS the matrices are in COLUMN major and in C++
 the matrices are in row major.
 so GpMatrix = Matrix'

 Note 1: This matrix is supposed to mimic a matrix on the device. 
 It contains functions that you can usually perform on a matrix.

 For operations on gpMatrix, refer gpMatrix_util.c{u,uh}
 For kernels refer to gpMatrixkernels. 

 */
class GpMatrix
{ 
 private:
  float *_deviceData; // device data
  int _numRows, _numCols;
  int _n_elem;
  bool _isTrans;
  int stride; 
  void _initGpMatrix(int numRows, int numCols, int stride, bool isTrans);
  static void cuBlaserrcheck(const char *msg) {
  	 	fprintf(stderr, msg);
  	 	exit(1);

  }
  
  

public:
   GpMatrix();
   GpMatrix(float *dev_data, int numRows, int numCols,bool isTrans);
   ~GpMatrix();
   static int getDeviceID();

   inline char getTransChar() {
  	 return _isTrans ? 'n':'t';
  }

   bool checkeqDims(const GpMatrix &mat) const {
   	return mat.getnumRows() == _numRows && mat.getnumCols() == _numCols;
   }
  
   int getnumRows() const {
   	 return _numRows;
   }

   int getnumCols() const {
   	  return _numCols;
   }


   int getnumElements() const {
   	 return _n_elem;
   }

   float *getDevData() const {
   	return _deviceData;
   }

   float* getoneCell(int i, int j) const{
       if(_isTrans)
       	return &_deviceData[j*_numRows+i];
       else
       	return &_deviceData[i*_numRows+j];
   }

   int getStride() const {
     return stride;
   }

   int getLeadingDim() const {
   	 return _isTrans ? _numRows : _numCols;
   }
   
   int getFollowingDim() const {
     return _isTrans? _numCols: _numRows;   
 }

 bool checkTrans() const {
 	return _isTrans;
 }
  
  bool checkContiguous() const {
  	return stride == getLeadingDim() || getFollowingDim() == 1; // for vectors. 
  }
   // void makeTrans(bool trans){
   // 	 if (trans != _isTrans)
   // 	 {  
   // 	 	trans = isTrans;
   // 	 }
   // }
void checkEqual(const GpMatrix &a, const GpMatrix &b) const; // check if matrices are equal
void resize(int Rows, int Cols); // resize the matrix according to the given dimensions.
void matCheckBounds(int numRows, int numCols) const;

bool checkContiguous(const GpMatrix &mat); // check if a GpMatrix is continguous.
// GpMatrix & sliceRow(int rowStart, int rowEnd) const; 
// GpMatrix & sliceCol(int colStart, int colEnd) const;
// GpMatrix & slice(int rowStart, int colStart, int rowEnd, int colEnd) const; // matrix slice operations. Return a new GpMatrix after slice.
// GpMatrix & reshape(int Rows, int Cols); // reshape the matrix acc to the args

void transposeMat(GpMatrix &tgt); // return the transpose of the matrix.
void printShape(GpMatrix &mat); // print the shape of the Matrix.
void addProduct(const GpMatrix &a, GpMatrix &b, float scaleThis, float scaleab); 
void add(GpMatrix &a, float scaleA, GpMatrix &b, float scaleB, GpMatrix &tgt);
void add(GpMatrix &b,float scale);
void subtract(GpMatrix &b, float scaleB, GpMatrix &tgt);
void subtract(GpMatrix &b, float scale);
void addVector(GpMatrix &vec, float scalevec, GpMatrix &tgt);
void addVector(GpMatrix &vec, float scale);
void addProduct(const GpMatrix &a, const GpMatrix &b, float scaleAB, float scaleC);
void RightMult(const GpMatrix &b, float scaleAB, GpMatrix &tgt);
void RightMult(const GpMatrix &b,float scale);
void RightMult(const GpMatrix &b, GpMatrix &tgt);
void elemWiseMult(GpMatrix &b, GpMatrix &tgt);
void elemWiseDivide(GpMatrix &b, GpMatrix &tgt);
void elemWiseDivide(GpMatrix &b);

void printMat(int numRows, int numCols);
// void exp(GpMatrix &b){
// 	return exp(b.getDevData());
// }

};


#endif