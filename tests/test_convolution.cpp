#include <iostream>
#include "../include/im2col.cuh"
#include "../include/cudaMatrix.hpp"
#include "../include/convcudamatrix.hpp"


int main(void)
{
	 float data1[] = {    1,1,2,0,1,
					      2,0,1,1,2,
					      0,2,2,1,0,
					      0,0,1,0,1,
					      0,2,0,0,1,
					      0,1,2,0,0,
					      2,2,1,0,2,
					      2,1,2,2,0,
					      1,0,1,0,1,
					      0,0,1,0,1,
					      1,0,1,2,2,
					      0,1,0,1,0,
					      0,2,2,1,1,
					      2,1,1,2,0,
					      2,1,0,1,2,
					  	  1,1,2,0,1,
					      2,0,1,1,2,
					      0,2,2,1,0,
					      0,0,1,0,1,
					      0,2,0,0,1,
					      0,1,2,0,0,
					      2,2,1,0,2,
					      2,1,2,2,0,
					      1,0,1,0,1,
					      0,0,1,0,1,
					      1,0,1,2,2,
					      0,1,0,1,0,
					      0,2,2,1,1,
					      2,1,1,2,0,
					      2,1,0,1,2};

	float filter[] = { -1, 0,-1,
					    0, 1, 0,
					    0,-1, 0,
					    0, 1, 0,
					    1, 0, 1,
					    0,-1, 1,
					    1,-1, 1,
					    0, 0, 1,
					    0,-1, 0,
					    1,-1,-1,
					    0,-1, 1,
					    0,-1, 1,
					    1, 1, 0,
					    1, 0, 0,
					    0, 1, 1,
					    1, 0,-1,
					   -1, 1, 1,
					    0,-1, 0 };
    int channels = 3;
    int stride = 2;
    int kern_sz=3;
    int height = 30;
    int width = 5;
    int col_width = kern_sz*kern_sz*channels;
    int col_height = ((height-kern_sz/2)/stride)*((width-kern_sz/2)/stride);
    int pad = 1;
    int bs = 4;

    cudaConvMatrix cv1(data1,2,5,5,3);
    cudaConvMatrix fl(filter,3,3,3,2);
    cudaConvMatrix tgt(2,5,5,3);
    cv1.fwd_pass_convolution_gpu(fl,tgt);
    float * result;
    result = new float[2*5*5*2];
    tgt.convgetData(result);
    for (int i = 0; i < 2; i ++) {
    	for (int m = 0; m < 5; m++) {
    		for (int n = 0; n < 5; n++ ) {
    			for (int k = 0; k < 3; k ++) {
    				std::cout << result(i,m,n,k) << " "; 
    			}
    		}
    	}
    }
     // float *col = new float[col_width*col_height];
    }
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