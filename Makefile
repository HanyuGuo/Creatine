objects: cudaMat.o test_cudaMat.o cudaKern.o

all:$(objects)
	nvcc $(objects) -o cudatest

%.o:cudaMatrix.cpp test_cudaMat.cpp cudaKernels.cu
	nvcc -x cu -I . -dc $< -o $@

clean:
	rm -rf *.o cudatest
