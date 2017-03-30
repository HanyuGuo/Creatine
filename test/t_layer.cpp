#include "../include/layers/layer.hpp"
#include <iostream>

using namespace std;

int main(void) {
	double * input = new double[3];
	for (int i=0; i<3; i++)
		input[i] = i;
	ip_layer input_layer(3);
	input_layer.feed(input);
	// input_layer.peek();
	// Matrix* Matrix_t = input_layer.getFprop();

	fc_layer l1(3,3);
	l1.feed(input_layer.getFprop());
	Matrix* Matrix_t = l1.getFprop();
	l1.forward(PASS_TRAIN);
	for (int i=0; i<3; i++) 
		cout << Matrix_t-> getCell(i,0);
	return 0;
}