#ifndef IM2_COL_CUH
#define IM2_COL_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col);
void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
