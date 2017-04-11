##ifndef CUDA_MATRIX_HPP_
#define CUDA_MATRIX_HPP_

class cudaMatrix {
private:
  float *devData;
  int numCols;
  int numRows;
  int numElems;
  void _init(float *data, int numRows, int numCols);

public:
  cudaMatrix(int numRows, int numCols);
  cudaMatrix(float *data, int numRows, int numCols);
  virtual ~cudaMatrix();
  int getNumRows() const {return numRows};
  int getnumCols() const {return numCols};
  void setDeviceData(const float *data); // set device Data;
  void getDeviceData(const float *hdata); // get device data in host pointer.
  void cudaAdd(float *a, float *b, int numRows, int numCols); // Matrix addition kernel.
  void cudaAdd(float *a, float *b);
};



#endif
