all:cudaMatrix.o test_cudaMat.o cudaKernel.o Activations.o
	nvcc cudaMatrix.o test_cudaMat.o cudaKernel.o Activations.o -o cudaTest -lcudart -lcuda

cudaMatrix.o: src/cudaMatrix.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

test_cudaMat.o:tests/test_cudaMat.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

cudaKernel.o:src/cudaKernels.cu
	nvcc -c $< -o $@

Activations.o:src/layer_kernels/Activations.cu
	nvcc -c $< -o $@

clean:
	rm *.o
