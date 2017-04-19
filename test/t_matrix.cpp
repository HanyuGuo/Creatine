#include "../include/matrix.hpp"
#include <iostream>

using namespace std;

int main(void){
	float * setA = new float[10];
	float * setB = new float[10];
	float * setC = new float[9];
	float * setD = new float[9];
	for (int i=0; i<10; i++)
		setA[i] = i;
	for (int i=0; i<5; i++)
		setB[i] = 0;
	for (int i=0; i<5; i++)
		setD[i] = 5;
	// Matrix A();
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl; 
	// Matrix A(rowA, colA);
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl; 
	Matrix *A = new Matrix(setA, 2, 5);
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl;
	// for (int i = 0; i< A.getNumElements(); i++)
	// 	cout << A(i/A.getNumCols(), i%A.getNumCols()) << endl;
	Matrix * B = new Matrix(setB, 2, 5);
	Matrix * C = new Matrix(setC, 2 ,5);
	Matrix * D = new Matrix(setD, 3 ,3);
	Matrix * BB = new Matrix(setD, 1 ,5);
	// float * reduce_sum = new float;
	// A.reluGrads(B, 0, C);
	// A.add(A, 0.1);
	// A.reduce_sum(*reduce_sum);
	A -> exp(-1,*B);
	// A -> rightMult(*BB, *C, true, false);
	for (int i = 0; i< B->getNumRows(); i++){
		for (int j = 0; j< B->getNumCols(); j++) {
			cout << " " << B -> getCell(i, j)<< " ";
		}
		cout << endl;
	}
	cout << (*B).max() << endl;
	// cout << "reduce_sum: "<< *reduce_sum<<endl;
	// // cout << "Does A has same dim as B: " << A.sameDim(B) << endl;
	// // cout << "Does A has same dim as C: " << A.sameDim(C) << endl;

	// Matrix * test0 = new Matrix(1,1);
	// Matrix * test = new Matrix(*test0);
	// delete test;

	return 0;
}