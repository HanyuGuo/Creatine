#include "../include/matrix.hpp"
#include <iostream>

using namespace std;

int main(void){
	double * setA = new double[9];
	double * setB = new double[9];
	double * setC = new double[9];
	double * setD = new double[9];
	for (int i=0; i<9; i++)
		setA[i] = i;
	for (int i=0; i<9; i++)
		setB[i] = 9 - i;
	int64 rowA = 3, colA = 3;
	for (int i=0; i<9; i++)
		setD[i] = 5;
	// Matrix A();
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl; 
	// Matrix A(rowA, colA);
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl; 
	Matrix A(setA, rowA, colA);
	// cout << "Row: " << A.getNumRows() << " Col: " << A.getNumCols() << " Elemts: " << A.getNumElements() << endl;
	// for (int i = 0; i< A.getNumElements(); i++)
	// 	cout << A(i/A.getNumCols(), i%A.getNumCols()) << endl;
	Matrix B(setB, 3, 3);
	Matrix C(setC, 3 ,3);
	Matrix D(setD, 3 ,3);
	double * reduce_sum = new double;
	A.add(A, 2);
	// A.reduce_sum(*reduce_sum);
	for (int i = 0; i< A.getNumRows(); i++){
		for (int j = 0; j< A.getNumCols(); j++) {
			cout << " " << A(i, j)<< " ";
		}
		cout << endl;
	}

	cout << "reduce_sum: "<< *reduce_sum<<endl;
	// cout << "Does A has same dim as B: " << A.sameDim(B) << endl;
	// cout << "Does A has same dim as C: " << A.sameDim(C) << endl;

	Matrix * test0 = new Matrix(1,1);
	Matrix * test = new Matrix(*test0);
	delete test;

	return 0;
}