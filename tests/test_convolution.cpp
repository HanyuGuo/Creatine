#include <iostream>
#include "../include/im2col.cuh"

int main(void)
{
	 float data1[] = {1,1,2,0,1,
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

	float filter[] = {-1, 0,-1,
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
    float *col = new float[col_width*col_height];
    im2col_gpu(data1,channels,height,width,kern_sz,stride,pad,col_height,col_width,col);

	return 0;
}