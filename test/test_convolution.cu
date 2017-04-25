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
    // int channels = 3;
    // int stride = 2;
    // int kern_sz=3;
    // int height = 30;
    // int width = 5;
    // int col_width = kern_sz*kern_sz*channels;
    // int col_height = ((height-kern_sz/2)/stride)*((width-kern_sz/2)/stride);
    // int pad = 1;
    // int bs = 4;

    cudaConvMatrix cv1(data1,4,4,2,2);
    cudaConvMatrix fl(filter,2,2,2,1);
    cudaConvMatrix tgt(4,4,2,1);
    cv1.convolve(fl, true, 1, tgt);

    // tgt.device2host();
    // for (int i = 0; i < 4; i ++) {
    // 	for (int m = 0; m < 4; m++) {
    // 		for (int n = 0; n < 2; n++ ) {
    // 			for (int k = 0; k < 1; k ++) {
    // 				std::cout << tgt(i,m,n,k) << " "; 
    // 			}
    // 			std::cout << "\n"; 
    // 		}
    // 		std::cout << "\n"; 
    // 	}
    // 	std::cout << "\n"; 
    // }
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