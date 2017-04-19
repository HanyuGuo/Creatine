#include "../../include/layers/layer.hpp"


void ip_layer::_init(int64 bs, int64 input_s, bool GPU) {
  _GPU = GPU;
  assert(input_s >= 0);
  if (GPU){
    _cudaInput = new cudaMatrix(bs, input_s);
    _cudaOutput = _cudaInput;
  }
  else {
    _input = new Matrix(bs, input_s);
    _output = _input;
    _parGrads = new Matrix(bs, input_s);
  }

}

ip_layer::ip_layer() {
  _init(0, 0, false);
}

ip_layer::ip_layer(int64 bs, int64 input_s) {
  _init(bs, input_s, false);
}

ip_layer::ip_layer(int64 bs, int64 input_s, bool GPU) {
  _init(bs, input_s, GPU);
}


void ip_layer::feed(float* input, int elems) {
  if(_GPU)
    _cudaInput -> setDeviceData(input, elems);
  else
    _input -> assign(input);
}

void ip_layer::forward(PASS_TYPE pass_type) {
  
}

void ip_layer::backward(PASS_TYPE pass_type) {

}

Matrix* ip_layer::getFprop() const {
  return _output;
}

cudaMatrix* ip_layer::getFprop(bool GPU) const {
  return _cudaOutput;
}

Matrix* ip_layer::getBprop() const {
  return _parGrads;
}

ip_layer::~ip_layer() {
 // delete _input;
  if (_GPU)
    delete _cudaInput;
  else
    delete _parGrads;
}

/*-------------------------fc layer-------------------------*/

void fc_layer::_init(int64 bs, int64 input_s, int64 output_s, bool biasThis, bool GPU) {
  _GPU = GPU;
  _bs = bs; 
  _input_s = input_s;
  _output_s = output_s;
  assert(input_s >= 0 && output_s >=0);
  if (_GPU){
    _input = new Matrix(bs, input_s);
    _output = new Matrix(bs, output_s);
    weight_init = new float[input_s * output_s];
    bias_init = new float[output_s];
    for (int i=0; i< input_s * output_s; i++) {
      weight_init[i] = ((float) rand() / (RAND_MAX)) + 1;
    }
    for (int i=0; i< output_s; i++) {
      bias_init[i] = ((float) rand() / (RAND_MAX)) + 1;
    }
    _weight = new Matrix(weight_init, input_s, output_s);
    _bias = new Matrix(bias_init, 1, output_s);
    // _parGrads = new Matrix(1, input_s);
    // _parGrads_w = new Matrix(input_s, output_s);
    // _parGrads_b = new Matrix(1, output_s);
  }
  else {
    _cudaInput = new cudaMatrix(bs, input_s);
    _cudaOutput = new cudaMatrix(bs, output_s);
    weight_init = new float[input_s * output_s];
    bias_init = new float[output_s];
    for (int i=0; i< input_s * output_s; i++) {
      weight_init[i] = ((float) rand() / (RAND_MAX)) + 1;
    }
    for (int i=0; i< output_s; i++) {
      bias_init[i] = ((float) rand() / (RAND_MAX)) + 1;
    }
    _cudaWeight = new cudaMatrix(weight_init, input_s, output_s);
    _cudaBias = new cudaMatrix(bias_init, 1, output_s);
  }

}

fc_layer::fc_layer() {
  _init(0, 0, 0,true, false);
}

fc_layer::fc_layer(int64 bs, int64 input_s, int64 output_s) {
  _init(bs, input_s, output_s, true, false);
}

fc_layer::fc_layer(int64 bs, int64 input_s, int64 output_s,  bool biasThis) {
  _init(bs, input_s, output_s, biasThis, false);
}

fc_layer::fc_layer(int64 bs, int64 input_s, int64 output_s,  bool biasThis, bool GPU) {
  _init(bs, input_s, output_s, biasThis, GPU);
}

void fc_layer::loadW(const char* path) {
  // load(path, weight_init);
  // if(_GPU) {
  //   _cudaWeight -> setDeviceData(weight_init, _input_s*_output_s);
  // }
  // else {
  //   _weight -> assign(weight_init);
  // }

}
void fc_layer::loadB(const char* path) {
  // load(path, bias_init);
  // if(_GPU) {
  //   _cudaBias -> setDeviceData(bias_init, 1*_output_s);
  // }
  // else {
  //   _bias -> assign(bias_init);
  // }

}

void fc_layer::feed(Matrix* input) {
	_input = input; 
}

void fc_layer::feed(cudaMatrix* input) {
  _cudaInput = input;
}

void  fc_layer::feedGrad(Matrix* prev_parGrads) {
  // _prev_parGrads = prev_parGrads;
}

//x*w+b
void fc_layer::forward(PASS_TYPE pass_type) {
  if (_GPU) {
    cudaMatrix tempProduct(_bs, _output_s);
    // _cudaInput -> gemm_ongpu(false, false, *_cudaWeight, 1, 1, tempProduct);
    // tempProduct.cudaAdd(tempProduct, *_cudaOutput);
  }
  else {
    _input -> rightMultPlus(*_weight, *_bias, *_output);    
  }

}




void fc_layer::calcParGrads(PASS_TYPE pass_type) {
  // _input -> rightMult(*_prev_parGrads, *_parGrads_w, true, false);
  // _parGrads_b -> copyTo(*_prev_parGrads);
}

//
void fc_layer::backward(PASS_TYPE pass_type) {
  // _prev_parGrads -> rightMult(*_weight, *_parGrads, false, true);


}

void fc_layer::applyGrads(float learningRate) {
  // _weight -> add(*_parGrads_w, -learningRate);
  // _bias -> add(*_parGrads_b, -learningRate);
}

Matrix* fc_layer::getFprop() const {
  return _output;
}

// Matrix* fc_layer::getBprop() const {
//   return _parGrads;
// }

fc_layer::~fc_layer() {
  if(_GPU) {
    delete _cudaOutput;
    delete _cudaWeight;
    delete _cudaBias;
  }
  else {
    delete _output;
    delete _weight;
    delete _bias;
  }
  // delete _input;

  // delete _parGrads;
  // delete _parGrads_w;
  // delete _parGrads_b;
}

/*-------------------------relu layer-------------------------*/
void relu_layer::_init(int64 input_s, int64 bs) {
  assert(input_s >= 0);
  _input = new Matrix(bs, input_s);
  _output = new Matrix(bs, input_s);
  _parGrads = new Matrix(bs, input_s);
  _tmp_exp = new Matrix(bs, input_s);
  _tmp_sum = new Matrix(bs, input_s);
}

relu_layer::relu_layer() {
  _init(0, 0);
}

relu_layer::relu_layer(int64 input_s, int64 bs) {
  _init(input_s, bs);
}

void relu_layer::feed(Matrix* input) {
  _input = input; 
}

void relu_layer::feedGrad(Matrix* prev_parGrads) {
  _prev_parGrads = prev_parGrads;
}

void relu_layer::forward(PASS_TYPE pass_type) {
  // _input -> exp(*_tmp_exp);
  // using namespace std;
  // cout << _tmp_exp -> getCell(0,0) << " " << _tmp_exp -> getCell(0,1) << endl;
  // _tmp_exp -> add(1, *_tmp_sum);
  // _tmp_sum -> ln(*_output);
  _input -> max(0, *_output);


}

// delta = 1/(1+exp(-x))
void relu_layer::backward(PASS_TYPE pass_type) {
  // Matrix tmp_NegExp(*_input);
  // Matrix tmp_sum(*_input);
  // Matrix tmp_grads(*_input);
  // _input -> exp(-1, tmp_NegExp);
  // tmp_NegExp.add(1, tmp_sum);
  // tmp_sum.eltwiseScaleDivideByThis(1, tmp_grads);
  // tmp_grads.dot(*_prev_parGrads, *_parGrads);
  _prev_parGrads -> reluGrads(*_input, *_parGrads);

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


/*-------------------------sigmoid layer-------------------------*/
void sigmoid::_init(int64 input_s, int64 bs) {
  assert(input_s >= 0);
  _input = new Matrix(bs, input_s);
  _output = new Matrix(bs, input_s);
  _parGrads = new Matrix(bs, input_s);
  _tmp_exp = new Matrix(bs, input_s);
  _tmp_sum = new Matrix(bs, input_s);
}

sigmoid::sigmoid() {
  _init(0, 0);
}

sigmoid::sigmoid(int64 input_s, int64 bs) {
  _init(input_s, bs);
}

void sigmoid::feed(Matrix* input) {
  _input = input; 
}

void sigmoid::feedGrad(Matrix* prev_parGrads) {
  _prev_parGrads = prev_parGrads;
}

void sigmoid::forward(PASS_TYPE pass_type) {
  Matrix tmp_NegExp(*_input);
  Matrix tmp_sum(*_input);
  Matrix tmp_grads(*_input);
  _input -> exp(-1, tmp_NegExp);
  tmp_NegExp.add(1, tmp_sum);
  tmp_sum.eltwiseScaleDivideByThis(1, *_output);


}


void sigmoid::backward(PASS_TYPE pass_type) {
  _prev_parGrads -> sigmoidGrads(*_input, *_parGrads);

}

Matrix* sigmoid::getFprop() const {
  return _output;
}

Matrix* sigmoid::getBprop() const {
  return _parGrads;
}

sigmoid::~sigmoid() {
  delete _output;
  delete _parGrads;
  delete _tmp_exp;
  delete _tmp_sum;
}




/*-------------------------conv2d layer-------------------------*/
void conv2d_layer::_init(int64 batch, int64 in_height, int64 in_width, 
          int64 filter_height, int64 filter_width, int64 in_channels,
          int64 out_channels, int64 stride, bool GPU) {
  _GPU = GPU;
  if(_GPU) {

  }
  else {

  }

}

conv2d_layer::conv2d_layer() {

}

conv2d_layer::conv2d_layer(int64 batch, int64 in_height, int64 in_width, 
          int64 filter_height, int64 filter_width, int64 in_channels,
          int64 out_channels, int64 stride) {
  
}

conv2d_layer::conv2d_layer(int64 batch, int64 in_height, int64 in_width, 
          int64 filter_height, int64 filter_width, int64 in_channels,
          int64 out_channels, int64 stride, bool GPU)  {
  
}

void conv2d_layer::loadW(const char* path)  {
  _GPU = GPU;
  if(_GPU) {

  }
  else {

  }
}



void conv2d_layer::feed(Matrix* input)  {

}

void conv2d_layer::feed(cudaMatrix* input)  {

}

void conv2d_layer::forward(PASS_TYPE pass_type) {
  _GPU = GPU;
  if(_GPU) {

  }
  else {

  }
}

Matrix* conv2d_layer::getFprop() const  {
  
}

conv2d_layer::~conv2d_layer()  {
  
}





/*-------------------------softmax layer-------------------------*/

void softmax_layer::_init(int64 input_s, int64 bs){
  assert(input_s >= 0);
  _input = new Matrix(bs, input_s);
  _output = new Matrix(bs, input_s);
  _parGrads = new Matrix(bs, input_s);
  _tmp_exp = new Matrix(bs,input_s);
  _tmp_exp_sum = new Matrix(bs,1);
}

softmax_layer::softmax_layer() {
  _init(0,0);
}

softmax_layer::softmax_layer(int64 input_s, int64 bs) {
  _init(input_s, bs);
}

void softmax_layer::feed(Matrix* input) {
  _input = input; 
}



void softmax_layer::forward(PASS_TYPE pass_type) {
  _input -> exp(*_tmp_exp);
  // using namespace std;
  // cout << "inner softmax: " << (*_input)(0,0) << " " <<  (*_input)(0,1) << endl;
  _tmp_exp -> reduce_sum(*_tmp_exp_sum);
  _tmp_exp -> eltwiseDivide(*_tmp_exp_sum, *_output);
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
// void cross_entropy::_init(int64 input_s) {
//   assert(input_s >= 0);
//   _logit = new Matrix(1, input_s);
//   _label = new Matrix(1, input_s);
//   _output = new float;
//   _parGrads = new Matrix(1, input_s);
// }

// cross_entropy::cross_entropy() {
//   _init(0);
// }

// cross_entropy::cross_entropy(int64 input_s) {
//   _init(input_s);
// }

// void cross_entropy::feed(Matrix* logit, Matrix* label) {
//   _logit = logit;
//   _label = label;
// }

// // ce = -1/N(label_i * log(logit_i) + (1 - label_i) * log(1 - logit_i))
// void cross_entropy::forward(PASS_TYPE pass_type) {
//   int64 N = _logit -> getNumCols();
//   Matrix tmp_log1(*_logit);
//   Matrix tmp_log2(*_logit);
//   Matrix tmp_dot1(*_logit);
//   Matrix tmp_dot2(*_logit);
//   float tmp_reduce_sum1, tmp_reduce_sum2;
//   Matrix tmp_subtract1(*_logit);
//   Matrix tmp_subtract2(*_logit);
//   Matrix tmp_sum(*_logit);

//   _logit -> log(tmp_log1);
//   _label -> dot(tmp_log1, tmp_dot1);
//   tmp_dot1.reduce_sum(tmp_reduce_sum1);
//   _label -> subtracted(1, tmp_subtract1);
//   _logit -> subtracted(1, tmp_subtract2);
//   tmp_subtract2.log(tmp_log2);
//   tmp_subtract1.dot(tmp_log2, tmp_dot2);
//   tmp_dot2.reduce_sum(tmp_reduce_sum2);
//   *_output = -(tmp_reduce_sum1 + tmp_reduce_sum2) / float(N);
// }

// // delta(ce)/delta(l) = logit - label
// void cross_entropy::backward(PASS_TYPE pass_type) {
//   _logit -> subtract(*_label, *_parGrads);
// }

// float* cross_entropy::getFprop() const {
//   return _output;
// }

// Matrix* cross_entropy::getBprop() const {
//   return _parGrads;
// }

// cross_entropy::~cross_entropy() {
//   delete _output;
//   delete _parGrads;
// }