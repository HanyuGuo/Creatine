#ifndef COL2IM_H
#define COL2IM_H

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im);

void col2im_ongpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);

#endif