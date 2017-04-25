#ifndef LAYER_H
#define LAYER_H
#include "../activations.cuh"
#include "../convMatrix.hpp"
#include "../cudaMatrix.cuh"
#include "../matrix.hpp"
#include "../utils.hpp"
#include <string>
#include <map>
#include <assert.h>
#include <iostream>


class ip_layer {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;
  bool _GPU;
  convMatrix* _convInput;
  convMatrix* _convOutput;
  void _init(int bs, int input_s, bool GPU);
  void _init(int D0, int D1, int D2, int D3, bool GPU);
  bool _fourD;
public:
  ip_layer();
  ip_layer(int bs, int input_s);
  ip_layer(int bs, int input_s, bool GPU);
  ip_layer(int D0, int D1, int D2, int D3);
  ip_layer(int D0, int D1, int D2, int D3, bool GPU);
  void feed(float* input, int elems);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  convMatrix* getConvFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  Matrix* getBprop() const;
  ~ip_layer();
};

class fc_layer {
protected:
  int _bs;
  int _input_s;
  int _output_s;
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  float* weight_init;
  float* bias_init;
  Matrix* _weight;
  Matrix* _bias;
  // Matrix* _prev_parGrads;
  // Matrix* _parGrads;
  // Matrix* _parGrads_w; // partial gradients
  // Matrix* _parGrads_b;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;   // a pointer so that could be used for later layers
  cudaMatrix* _cudaWeight;
  cudaMatrix* _cudaBias;
  bool _GPU;

  void _init(int bs, int input_s, int output_s, bool biasThis, bool GPU);
public:
  fc_layer();
  fc_layer(int bs, int input_s, int output_s);
  fc_layer(int bs, int input_s, int output_s, bool biasThis);
  fc_layer(int bs, int input_s, int output_s, bool biasThis, bool GPU);
  void loadW(const char* path);
  void loadB(const char* path);
  void feed(Matrix* input);
  void feed(cudaMatrix* input);
  void feedGrad(Matrix* prev_parGrads);
  void forward(PASS_TYPE pass_type);
  void calcParGrads(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  void applyGrads(float learningRate);
  Matrix* getFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  // Matrix* getBprop() const;
  ~fc_layer();
};


class relu_layer {
protected:
  bool _GPU;
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  Matrix* _prev_parGrads;
  Matrix* _tmp_exp;
  Matrix* _tmp_sum;
  convMatrix* _convInput;
  convMatrix* _convOutput;
  Matrix* _flattenOutput;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;

  bool _fourD;
  void _init(int bs, int input_s, bool GPU);
  void _init(int D0, int D1, int D2, int D3, bool GPU);

public:
  relu_layer();
  relu_layer(int bs, int input_s, bool GPU);
  relu_layer(int D0, int D1, int D2, int D3);
  relu_layer(int D0, int D1, int D2, int D3, bool GPU);
  void feed(Matrix* input);
  void feed(convMatrix* input);
  void feed(cudaMatrix* input);
  void feedGrad(Matrix* prev_parGrads);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  convMatrix* getConvFprop() const;
  Matrix* getFlattenFprop() const;
  Matrix* getBprop() const;
  ~relu_layer();
};


class sigmoid {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  Matrix* _prev_parGrads;
  Matrix* _tmp_exp;
  Matrix* _tmp_sum;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;   // a pointer so that could be used for later layers
  cudaMatrix* _cudaWeight;
  cudaMatrix* _cudaBias;
  bool _GPU;
  void _init(int input_s, int bs, bool GPU);
public:
  sigmoid();
  sigmoid(int input_s, int bs, bool GPU);
  void feed(Matrix* input);
  void feed(cudaMatrix* input);
  void feedGrad(Matrix* prev_parGrads);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  Matrix* getBprop() const;
  ~sigmoid();
};


class conv2d_layer {
protected:
  int _batch;
  int _in_height;
  int _in_width;
  int _filter_height;
  int _filter_width;
  int _in_channels;
  int _out_channels;
  int _stride;
  float* weight_init;
  convMatrix* _input;
  convMatrix* _weight;
  convMatrix* _output;
  Matrix* _flattenOutput;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;   // a pointer so that could be used for later layers
  cudaMatrix* _cudaWeight;
  bool _GPU;

  void _init(int batch, int in_height, int in_width, 
            int filter_height, int filter_width, int in_channels,
            int out_channels, int stride, bool GPU);
public:
  conv2d_layer();
  conv2d_layer(int batch, int in_height, int in_width, 
            int filter_height, int filter_width, int in_channels,
            int out_channels, int stride);
  conv2d_layer(int batch, int in_height, int in_width, 
            int filter_height, int filter_width, int in_channels,
            int out_channels, int stride, bool GPU);
  void loadW(const char* path);
  void feed(convMatrix* input);
  void feed(cudaMatrix* input);
  void forward(PASS_TYPE pass_type);
  convMatrix* getFprop() const;
  Matrix* getFlattenFprop() const;
  ~conv2d_layer();
};




class softmax_layer {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  Matrix * _tmp_exp;
  Matrix * _tmp_exp_sum;
  void _init(int bs, int input_s, bool GPU);
  bool _GPU;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;   // a pointer so that could be used for later layers
public:
  softmax_layer();
  softmax_layer(int bs, int input_s);
  softmax_layer(int bs, int input_s, bool GPU);
  void feed(Matrix* input);
  void feed(cudaMatrix* input);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  Matrix* getBprop() const;
  ~softmax_layer();
};

// class cross_entropy {
// protected:
//   Matrix* _logit;
//   Matrix* _label;
//   float* _output;   // a pointer so that could be used for later layers
//   Matrix* _parGrads; // partial gradients
//   void _init(int input_s);
// public:
//   cross_entropy();
//   cross_entropy(int input_s);
//   void feed(Matrix* logit, Matrix* label);
//   void forward(PASS_TYPE pass_type);
//   void backward(PASS_TYPE pass_type);
//   float* getFprop() const;
//   Matrix* getBprop() const;
//   ~cross_entropy();
// };


#endif

