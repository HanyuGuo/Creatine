#ifndef IM2_COL_CUH
#define IM2_COL_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void im2col_gpu_kernel(int bs, const float *ip, int height, int width, int kernel_sz, int pad, int stride, int col_height, int col_width, float *op);
void im2col_gpu(float *ip_img, const int channels, const int height, const int width ,const int kernel_sz,const int stride, const int pad, const int col_height, const int col_width, float *op);
#endif
