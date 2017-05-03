run:cudaMatrix.o main.o cudaKernel.o matrix.o utils.o layer.o convMatrix.o activations.o
	nvcc cudaMatrix.o cudaKernel.o main.o matrix.o utils.o layer.o convMatrix.o activations.o -o run -lcudart -lcuda -lcublas -lcblas

cudaMatrix.o: src/cudaMatrix.cu
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

main.o:test/main.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@



cudaKernel.o:src/cudaKernels.cu
	nvcc -c $< -o $@

activations.o:src/activations.cu
	nvcc -c $< -o $@


matrix.o: src/matrix.cpp
	nvcc -c $< -o $@

utils.o: src/utils.cpp 
	nvcc -std=c++11 -c $< -o $@

layer.o: src/layers/layer.cpp
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@

convMatrix.o: src/convMatrix.cpp
	nvcc -c $< -o $@

clean:
	rm *.o


