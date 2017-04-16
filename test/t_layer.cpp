#include "../include/layers/layer.hpp"
#include <iostream>

using namespace std;

int main(void) {
	double * input = new double[3];
	for (int i=0; i<3; i++)
		input[i] = 0.33;
	ip_layer * input_layer = new ip_layer(3);
	input_layer -> feed(input);
	// input_layer.peek();

	double * label = new double[3];
	for (int i=0; i<3; i++)
		label[i] = 0.33;
	ip_layer * label_layer = new ip_layer(3);
	label_layer -> feed(label);


	fc_layer * l1 = new fc_layer(3,3);
	l1->feed(input_layer->getFprop());
	Matrix* Matrix_t = l1->getFprop();
	l1->forward(PASS_TRAIN);
	cout << "fully connected layer" << endl;
	for (int i=0; i<3; i++) 
		cout << Matrix_t-> getCell(0,i) <<" ";
	cout << "\nPass" << endl;

	cout << "relu layer" << endl;
	relu_layer * relu = new relu_layer(3);
	relu->feed(l1->getFprop());
	Matrix* Matrix_relu = relu->getFprop();
	relu->forward(PASS_TRAIN);
	cout << "fully connected layer" << endl;
	for (int i=0; i<3; i++) 
		cout << Matrix_relu-> getCell(0,i) <<" " ;
	cout << "\nPass" << endl;

	cout << "softmax layer" << endl;
	softmax_layer * l2 = new softmax_layer(3);
	l2->feed(relu->getFprop());
	l2->forward(PASS_TRAIN);
	Matrix* Matrix_sftmx = l2->getFprop();
	for (int i=0; i<3; i++) 
		cout << Matrix_sftmx-> getCell(0,i)<<" ";
	cout << "\nPass" << endl;
	
    cout << "cross entropy" << endl;
	cross_entropy * cost  = new cross_entropy(3);
	cost -> feed(l2 -> getFprop(), label_layer -> getFprop());
	cost -> forward(PASS_TRAIN);
	double* ce = cost->getFprop();
	cout << *ce;
	cout << "\nPass" << endl;

	cout << "---------------------backward view--------------------" << endl;

	cost -> backward(PASS_TRAIN);
	Matrix* Matrix_b_cost = cost->getBprop();
	for (int i=0; i<3; i++) 
		cout << Matrix_b_cost-> getCell(0,i) << " ";
	cout << "\nPass" << endl;

	relu -> feedGrad(cost -> getBprop());
	relu -> backward(PASS_TRAIN);
	l1 -> feedGrad(relu -> getBprop());
	l1 -> backward(PASS_TRAIN);	
	l1 -> calcParGrads(PASS_TRAIN);
	l1 -> applyGrads(0.1);
	// delete l1;
	// delete input_layer;
	// delete l2;




	return 0;
}