all:cudaMatrix.o test_cudaMat.o cudaKernel.o
	nvcc cudaMatrix.o test_cudaMat.o cudaKernel.o -o cudaTest -lcudart -lcuda

cudaMatrix.o: src/cudaMatrix.cpp
	nvcc -x cu -I. -dc $< -o $@

test_cudaMat.o: test_cudaMat.cpp
	nvcc -x cu -I. -dc $< -o $@

cudaKernel.o:src/cudaKernels.cu
	nvcc $< -o $@
