#include "../include/layers/layer.hpp"
#include <iostream>

using namespace std;

int main(void) {
	double * input = new double[3];
	for (int i=0; i<3; i++)
		input[i] = i;
	ip_layer * input_layer = new ip_layer(3);
	input_layer -> feed(input);
	input_layer->feed(input);
	// input_layer.peek();


	fc_layer * l1 = new fc_layer(3,3);
	l1->feed(input_layer->getFprop());
	Matrix* Matrix_t = l1->getFprop();
	l1->forward(PASS_TRAIN);
	cout << "fully connected layer" << endl;
	for (int i=0; i<3; i++) 
		cout << Matrix_t-> getCell(i,0);
	cout << "\nPass" << endl;


	cout << "softmax layer" << endl;
	softmax_layer * l2 = new softmax_layer(3);
	l2->feed(l1->getFprop());
	l2->forward(PASS_TRAIN);
	Matrix* Matrix_sftmx = l2->getFprop();
	for (int i=0; i<3; i++) 
		cout << Matrix_sftmx-> getCell(i,0);
	cout << "\nPass" << endl;
	delete l1;
	delete input_layer;

	return 0;
}