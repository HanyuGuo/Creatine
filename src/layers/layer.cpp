#include "../../include/layers/layer.hpp"


void ip_layer::_init(int64 input_s) {
  assert(input_s >= 0);
  _input = new Matrix(1, input_s);
  _output = _input;
  _parGrads = new Matrix(1, input_s);
}

ip_layer::ip_layer() {
  _init(0);
}

ip_layer::ip_layer(int64 input_s) {
  _init(input_s);
}

void ip_layer::peek() {
  using namespace std;
  cout << "peek input layer: " <<_input -> getCell(0, 0) << endl;
}

void ip_layer::feed(double* input) {
  _input -> assign(input);
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
  _input = new Matrix(1, input_s);
  _output = new Matrix(1, output_s);
  weight_init = new double[input_s * output_s];
  bias_init = new double[output_s];
  for (int i=0; i<input_s * output_s; i++) {
  	weight_init[i] = 0.1;
  }
  for (int i=0; i<output_s; i++) {
  	bias_init[i] = 0.1;
  }
  _weight = new Matrix(weight_init, input_s, output_s);
  _bias = new Matrix(bias_init, 1, output_s);
  _parGrads = new Matrix(1, input_s);
  _parGrads_w = new Matrix(input_s, output_s);
  _parGrads_b = new Matrix(1, output_s);
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

void  fc_layer::feedGrad(Matrix* prev_parGrads) {
  _prev_parGrads = prev_parGrads;
}

void fc_layer::forward(PASS_TYPE pass_type) {
  // using namespace std;
  
  _input -> rightMultPlus(*_weight, *_bias, *_output);
  // cout << "input :"<< _output->getCell(0,0) << " "<< _output->getCell(0,1) << " "
  // << _output->getCell(0,2)<< " "<< _output->getCell(0,3) << " "<< _output->getCell(0,4)<<endl;
}


void fc_layer::calcParGrads(PASS_TYPE pass_type) {
  _input -> rightMult(*_prev_parGrads, *_parGrads_w, true, false);
  _parGrads_b = _prev_parGrads;
}

//
void fc_layer::backward(PASS_TYPE pass_type) {
  using namespace std;
  cout << "_weight:" << _weight->getNumRows() << _weight->getNumCols() << endl;
  cout << "_prev_parGrads:" << _prev_parGrads -> getNumRows() << _prev_parGrads->getNumCols() << endl;
  cout << "_parGrads:" << _parGrads -> getNumRows() << _parGrads ->getNumCols() << endl;
  // cout << "_weight:" << _weight->getCell(0,0) << _weight->getCell(0,1) << endl;
  // cout << "_prev_parGrads:" << _prev_parGrads -> getCell(0,0) << _prev_parGrads->getCell(0,1) << endl;
  // cout << "_parGrads:" << _parGrads -> getCell(0,0) << _parGrads ->getCell(0,1) << endl;

  _prev_parGrads -> rightMult(*_weight, *_parGrads, false, true);
}

void fc_layer::applyGrads(double learningRate) {
  _weight -> add(*_parGrads_w, -learningRate);
  _bias -> add(*_parGrads_b, -learningRate);
}

Matrix* fc_layer::getFprop() const {
  return _output;
}

Matrix* fc_layer::getBprop() const {
  return _parGrads;
}

fc_layer::~fc_layer() {
  // delete _input;
  delete _output;
  delete _weight;
  delete _bias;
  delete _parGrads_w;
  delete _parGrads_b;
  delete weight_init;
  delete bias_init;
}

/*-------------------------sigmoid layer-------------------------*/
void relu_layer::_init(int64 input_s) {
  assert(input_s >= 0);
  _input = new Matrix(1, input_s);
  _output = new Matrix(1, input_s);
  _parGrads = new Matrix(1, input_s);
  _tmp_exp = new Matrix(1, input_s);
  _tmp_sum = new Matrix(1, input_s);
}

relu_layer::relu_layer() {
  _init(0);
}

relu_layer::relu_layer(int64 input_s) {
  _init(input_s);
}

void relu_layer::feed(Matrix* input) {
  _input = input; 
}

void relu_layer::feedGrad(Matrix* prev_parGrads) {
  _prev_parGrads = prev_parGrads;
}

void relu_layer::forward(PASS_TYPE pass_type) {
  _input -> exp(*_tmp_exp);
  _tmp_exp -> add(1, *_tmp_sum);
  _tmp_sum -> ln(*_output);
}

// delta = 1/(1+exp(-x))
void relu_layer::backward(PASS_TYPE pass_type) {
  Matrix tmp_NegExp(*_input);
  Matrix tmp_sum(*_input);
  Matrix tmp_grads(*_input);
  _input -> exp(-1, tmp_NegExp);
  tmp_NegExp.add(1, tmp_sum);
  tmp_sum.eltwiseScaleDivideByThis(1, tmp_grads);

  tmp_grads.dot(*_prev_parGrads, *_parGrads);
}

Matrix* relu_layer::getFprop() const {
  return _output;
}

Matrix* relu_layer::getBprop() const {
  return _parGrads;
}

relu_layer::~relu_layer() {
  delete _output;
  delete _parGrads;
  delete _tmp_exp;
  delete _tmp_sum;
}




/*-------------------------softmax layer-------------------------*/

void softmax_layer::_init(int64 input_s){
  assert(input_s >= 0);
  _input = new Matrix(1, input_s);
  _output = new Matrix(1, input_s);
  _parGrads = new Matrix(1, input_s);
  _tmp_exp = new Matrix(1,input_s);
  _tmp_exp_sum = new double;
}

softmax_layer::softmax_layer() {
  _init(0);
}

softmax_layer::softmax_layer(int64 input_s) {
  _init(input_s);
}

void softmax_layer::feed(Matrix* input) {
  _input = input; 
}



void softmax_layer::forward(PASS_TYPE pass_type) {
  _input -> exp(*_tmp_exp);
  _tmp_exp -> reduce_sum(*_tmp_exp_sum);
  _tmp_exp -> eltwiseDivideByScale(*_tmp_exp_sum, *_output);
}


void softmax_layer::backward(PASS_TYPE pass_type) {

}

Matrix* softmax_layer::getFprop() const {
  return _output;
}

Matrix* softmax_layer::getBprop() const {
  return _parGrads;
}

softmax_layer::~softmax_layer() {
  // delete _input; //TODO: change _input delete
  delete _output;
  delete _parGrads;
  delete _tmp_exp_sum;
  delete _tmp_exp;
}


/*-------------------------cross entropu-------------------------*/
void cross_entropy::_init(int64 input_s) {
  assert(input_s >= 0);
  _logit = new Matrix(1, input_s);
  _label = new Matrix(1, input_s);
  _output = new double;
  _parGrads = new Matrix(1, input_s);
}

cross_entropy::cross_entropy() {
  _init(0);
}

cross_entropy::cross_entropy(int64 input_s) {
  _init(input_s);
}

void cross_entropy::feed(Matrix* logit, Matrix* label) {
  _logit = logit;
  _label = label;
}

// ce = -1/N(label_i * log(logit_i) + (1 - label_i) * log(1 - logit_i))
void cross_entropy::forward(PASS_TYPE pass_type) {
  int64 N = _logit -> getNumCols();
  Matrix tmp_log1(*_logit);
  Matrix tmp_log2(*_logit);
  Matrix tmp_dot1(*_logit);
  Matrix tmp_dot2(*_logit);
  double tmp_reduce_sum1, tmp_reduce_sum2;
  Matrix tmp_subtract1(*_logit);
  Matrix tmp_subtract2(*_logit);
  Matrix tmp_sum(*_logit);

  _logit -> log(tmp_log1);
  _label -> dot(tmp_log1, tmp_dot1);
  tmp_dot1.reduce_sum(tmp_reduce_sum1);
  _label -> subtracted(1, tmp_subtract1);
  _logit -> subtracted(1, tmp_subtract2);
  tmp_subtract2.log(tmp_log2);
  tmp_subtract1.dot(tmp_log2, tmp_dot2);
  tmp_dot2.reduce_sum(tmp_reduce_sum2);
  *_output = -(tmp_reduce_sum1 + tmp_reduce_sum2) / double(N);
}

// delta(ce)/delta(l) = logit - label
void cross_entropy::backward(PASS_TYPE pass_type) {
  _logit -> subtract(*_label, *_parGrads);
}

double* cross_entropy::getFprop() const {
  return _output;
}

Matrix* cross_entropy::getBprop() const {
  return _parGrads;
}

cross_entropy::~cross_entropy() {
  // delete _logit;
  // delete _label;
  // delete _output;
  // delete _parGrads;
}