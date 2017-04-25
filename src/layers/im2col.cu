#include <stdio.h>
#include "../../include/cudaMatrix.hpp"
// #include "../../include/convlayer.h"
#include "../../include/im2col.cuh"



//take patches from a 3d input image and convert put them in the output matrix.
// src: https://github.com/pjreddie/darknet/src/im2col.cu

__global__ void im2col_gpu_kernel(int bs, const float *ip,
                 int height,
                 int width,
                 int kernel_sz,
                 int pad,
                 int stride,
                 int col_height ,int col_width,
                 float *op) {

    for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < bs; idx += gridDim.x*blockDim.x) {
        int w_out = idx%col_width;
        int h_idx = idx/col_height;
        int h_out = h_idx%col_height;
        int h_in = h_out*stride - pad;
        int w_in = w_out*stride - pad;
        int ch_in = h_in/col_height;
        int ch_out = ch_in*kernel_sz*kernel_sz;
        float *op_col_ptr = op;
        op_col_ptr +=(ch_out*col_height + h_out)*col_width + w_out;
        const float *data_im_ptr = ip;
        data_im_ptr += (ch_in*height + h_in)*width + w_in;
        for (int i = 0; i < kernel_sz; ++i) {
          for(int j = 0; j<kernel_sz; ++j) {
            int h_x = h_in + i;
            int h_y = w_in + j;

            *op_col_ptr = (h_x > 0 && h_y >0 && h_x < width && h_y < height) ? data_im_ptr[i*width+j]:0;
            op_col_ptr += col_height*col_width;
          }
        }

    }

}



void im2col_gpu(float *ip_img, const int channels, const int height, const int width,
                const int kern_sz, const int stride, const int pad,const int col_height, const int col_width, float *op_col){
          cudaError_t err;
          int h_col = (height + 2*pad - kern_sz)/stride;
          int w_col = (height + 2*pad - kern_sz)/stride;
          int num_kernels = channels*h_col*w_col;
          int block = 512;
          int grid = (num_kernels+block -1)/block;
          im2col_gpu_kernel <<< grid, block >>> (num_kernels, ip_img, height, width, kern_sz,pad,stride,col_height,col_width,op_col);
          err = cudaGetLastError();
          printf("err: %d", err);
          if (err != cudaSuccess) {
            printf("can't launch im2col \n");
          }


}
