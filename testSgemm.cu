#include<cuda.h>
#include<cublas_v2.h>
#include <stdio.h>
int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}
int main(int argc, char const *argv[]) {
  int rowA, rowB, colA, colB,rowC, colC;
  rowA = 5;
  colA = 2;
  rowB = colA;
  colB = 3;
  rowC = rowA;
  colC = colB;
  float *ddata1, *ddata2, *resdata;
  float *data1 = new float[rowA*colA];
  float *data2 = new float[rowB*colB];
  float *res = new float[rowC*colC];

  for (int i = 0; i < rowA; ++i) {
    for (int j = 0; j < colA; j++) {
         data1[ci(i,j,colA)] = i;

    }
}

for (int i = 0; i < rowB; ++i) {
  for (int j = 0; j < colB; j++) {
       data2[ci(i,j,colB)] = i;

  }
}
cudaMalloc((void**)&ddata1,rowA*colA*sizeof(float));
cudaMalloc((void**)&ddata2,rowB*colB*sizeof(float));
cudaMalloc((void**)&resdata,rowC*colC*sizeof(float));
cudaMemcpy(ddata1, data1, rowA*colA*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(ddata2, data2, rowB*colB*sizeof(float),cudaMemcpyHostToDevice);
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f;
float beta = 0.0f;
cudaError_t err;
cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,colB,rowA,colA,&alpha,ddata2,colB,ddata1,colA,&beta,resdata,colB);
err = cudaGetLastError();
if (err != cudaSuccess) {
  printf("Can't perform Sgemm...\n");
}
cudaMemcpy(res, resdata, rowC*colC*sizeof(float),cudaMemcpyDeviceToHost);
for (int i = 0; i < rowC; i++) {
  for (int j = 0; j < colC; j++) {
       printf("%.2f ",res[ci(i,j,colC)]);

  }
   printf("\n");
}

  return 0;
}
