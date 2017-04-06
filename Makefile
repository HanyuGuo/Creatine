ROOT_DIR='$HOME/Software/Creatine'
INCLUDE_DIR='$ROOT_DIR/include'
SRC_DIR='$ROOT_DIR/src'

all:gpmat.cu.o
	g++ -o gpmtest -lcudart gpmat.o

gpmat.cu.o:gpmatrix.cu gpmatrix.cu
	nvcc -c $SRC_DIR/gpmatrix.cu -o $@