test:cudaMatrix.o test_convolution.o cudaKernel.o Activations.o im2col.o convMatrix.o
	nvcc cudaMatrix.o test_convolution.o cudaKernel.o Activations.o im2col.o convMatrix.o -o cudaTest -lcudart -lcuda -lcublas

cudaMatrix.o: src/cudaMatrix.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

test_convolution.o:tests/test_convolution.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

test_cudaMat.o:tests/test_cudaMat.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

cudaKernel.o:src/cudaKernels.cu
	nvcc -c $< -o $@

Activations.o:src/layer_kernels/Activations.cu
	nvcc -c $< -o $@

convMatrix.o:src/convMatrix.cpp 
	nvcc -c $< -o $@

im2col.o:src/layers/im2col.cu
	nvcc -c $< -o $@



clean:
	rm *.o
