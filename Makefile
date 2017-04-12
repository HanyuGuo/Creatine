all:cudaMatrix.o test_cudaMat.o cudaKernel.o
	nvcc cudaMatrix.o test_cudaMat.o cudaKernel.o -o cudaTest -lcudart -lcuda

cudaMatrix.o: cudaMatrix.cpp
	nvcc -x cu -I. -dc $< -o $@

test_cudaMat.o: test_cudaMat.cpp
	nvcc -x cu -I. -dc $< -o $@

cudaKernel.o:cudaKernels.cu
	nvcc $< -o $@
