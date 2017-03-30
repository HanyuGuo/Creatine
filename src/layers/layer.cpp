#include "../../include/layers/layer.hpp"


void ip_layer::_init(int64 input_s) {
  assert(input_s >= 0);
  _input = new Matrix(input_s, 1);
  _output = _input;
  _parGrads = new Matrix(input_s, 1);
}

ip_layer::ip_layer() {
  _init(0);
}

ip_layer::ip_layer(int64 input_s) {
  _init(input_s);
}

void ip_layer::peek() {
  using namespace std;
  cout << "peek input layer: " <<_input->getCell(0, 0) << endl;
}

void ip_layer::feed(double* input) {
  _input->assign(input);
}

void ip_layer::forward(PASS_TYPE pass_type) {
  
}

void ip_layer::backward(PASS_TYPE pass_type) {

}

Matrix* ip_layer::getFprop() const {
  return _output;
}

Matrix* ip_layer::getBprop() const {
  return _parGrads;
}

ip_layer::~ip_layer() {
 delete _input;
 delete _parGrads;
}
/*-------------------------fc layer-------------------------*/

void fc_layer::_init(int64 input_s, int64 output_s, bool biasThis) {
  assert(input_s >= 0 && output_s >=0);
  _input = new Matrix(input_s, 1);
  _output = new Matrix(output_s, 1);
  double* weight_init = new double[9];
  double* bias_init = new double[3];
  for (int i=0; i<9; i++) {
  	weight_init[i] = 1;
  }
  for (int i=0; i<3; i++) {
  	bias_init[i] = 1;
  }
  _weight = new Matrix(weight_init, input_s, output_s);
  _bias = new Matrix(bias_init, output_s, 1);
  _parGrads_w = new Matrix(input_s, output_s);
  _parGrads_b = new Matrix(output_s, 1);
}

fc_layer::fc_layer() {
  _init(0, 0, true);
}

fc_layer::fc_layer(int64 input_s, int64 output_s) {
  _init(input_s, output_s, true);
}

fc_layer::fc_layer(int64 input_s, int64 output_s, bool biasThis) {
  _init(input_s, output_s, biasThis);
}

void fc_layer::feed(Matrix* input) {
	_input = input; 
}

void fc_layer::forward(PASS_TYPE pass_type) {
  _weight->rightMultPlus(*_input, *_bias, *_output);
}
void fc_layer::backward(PASS_TYPE pass_type) {

}

Matrix* fc_layer::getFprop() {
  return _output;
}

Matrix* fc_layer::getBprop() {
  return _parGrads_w;
}

fc_layer::~fc_layer() {
  delete _input;
  delete _output;
  delete _weight;
  delete _bias;
  delete _parGrads_w;
  delete _parGrads_b;
}