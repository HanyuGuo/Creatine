#include <iostream>
#include "../include/im2col.cuh"
#include "../include/cudaMatrix.cuh"
#include "../include/convcudamatrix.cuh"


int main(void)
{
	 float data1[] = {    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

	float filter[] = { 0, 1, 2, 3, 4, 5, 6, 7};

	float data2[] = {0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.};


    cudaConvMatrix cv1(data1,2,4,4,2);
    cudaConvMatrix fl(filter,2,2,2,1);
    cudaConvMatrix tgt(data2, 2,4,4,1);
    cv1.convolve(fl, true, 1, tgt);

    float * result = new float[4*4*2];
    tgt.getDeviceData(result);
    for (int i = 0; i < 2; i ++) {
    	for (int m = 0; m < 4; m++) {
    		for (int n = 0; n < 4; n++ ) {
    			for (int k = 0; k < 1; k ++) {
    				std::cout << result[i*4*4*1 + m*4 + n+k] << " "; 
    			}
    			std::cout << "\n"; 
    		}
    		std::cout << "\n"; 
    	}
    	std::cout << "\n"; 
    }
     // float *col = new float[col_width*col_height];
    // float *ddata, *dcol;
    // cudaMalloc((void**)&ddata, height*width*sizeof(float));
    // cudaMalloc((void**)&dcol, col_height*col_width*sizeof(float));
    // cudaMemcpy(ddata,&data1,height*width*sizeof(float),cudaMemcpyHostToDevice);
    // im2col_gpu(ddata,channels,height,width,kern_sz,stride,pad,col_height,col_width,dcol);
    // cudaMemcpy(col,dcol,col_height*col_width*sizeof(float),cudaMemcpyDeviceToHost);
    // cudaFree(ddata);
    // cudaFree(dcol);    
	return 0;
}