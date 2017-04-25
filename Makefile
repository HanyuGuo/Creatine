test:cudaMatrix.o test_convolution.o cudaKernel.o activations.o im2col.o convcudamatrix.o col2im.o
	nvcc cudaMatrix.o test_convolution.o cudaKernel.o activations.o im2col.o convcudamatrix.o  col2im.o -o cudaTest -lcudart -lcuda -lcublas

cudaMatrix.o: src/cudaMatrix.cu
	nvcc  -arch=sm_35 -x cu -I. -dc $< -o $@

test_convolution.o:test/test_convolution.cu
	nvcc -arch=sm_35 -x cu -I. -dc $< -o $@


cudaKernel.o:src/cudaKernels.cu
	nvcc  -c $< -o $@

activations.o:src/activations.cu
	nvcc  -c $< -o $@

convcudamatrix.o:src/convcudamatrix.cu
	nvcc  -arch=sm_35 -x cu -I. -dc $< -o $@

im2col.o:src/im2col.cu
	nvcc -c $< -o $@

col2im.o:src/col2im.cu
	nvcc -c $< -o $@

clean:
	rm *.o
