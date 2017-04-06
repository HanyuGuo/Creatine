#ifndef GPMATRIX_KERNELS_CUH
#define GPMATRIX_KERNELS_CUH

#define ADD_BLOCK_WIDTH 16
#define ELEM_WISE_THX  16
#define ELEM_WISE_THY 16



__global__ void MatAdd(double *a, double *b, double *c,
					   int strideA, int strideB, int stridetgt) {
	int idx = blockIdx.x*ELEM_WISE_THX + threadIdx.x;
	int idy = blockIdx.y*ELEM_WISE_THY + threadIdx.y;
    
    if (idx < width && idy < height)
    {
    	c[idy*stridetgt+idx] = a[idy*strideA+idx]+b[idy*strideB+idx];
    }

}


__global__ void MatMul(double *a, double *b, double *c){
	int idx = blockIdx.x*ELEM_WISE_THX + threadIdx.x;
	int idy = blockIdx.y*ELEM_WISE_THY + threadIdx.y;
    
    if (idx < width && idy < height)
    {
    	c[idy*stridetgt+idx] = a[idy*strideA+idx]*b[idy*strideB+idx];
    }

}

__global__ void MatDivide(double *a, double *b, )





#endif