#ifndef LAYER_H
#define LAYER_H

#include "../cudaMatrix.cuh"
#include "../matrix.hpp"
#include "../utils.hpp"
#include <string>
#include <map>
#include <assert.h>
#include <iostream>
typedef long long int int64;


class ip_layer {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  void _init(int64 bs, int64 input_s, bool GPU);
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;
  bool _GPU;
public:
  ip_layer();
  ip_layer(int64 bs, int64 input_s);
  ip_layer(int64 bs, int64 input_s, bool GPU);
  void feed(float* input, int elems);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  cudaMatrix* getFprop(bool GPU) const;
  Matrix* getBprop() const;
  ~ip_layer();
};

class fc_layer {
protected:
  int64 _bs;
  int64 _input_s;
  int64 _output_s;
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

  void _init(int64 bs, int64 input_s, int64 output_s, bool biasThis, bool GPU);
public:
  fc_layer();
  fc_layer(int64 bs, int64 input_s, int64 output_s);
  fc_layer(int64 bs, int64 input_s, int64 output_s, bool biasThis);
  fc_layer(int64 bs, int64 input_s, int64 output_s, bool biasThis, bool GPU);
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
  // Matrix* getBprop() const;
  ~fc_layer();
};


class relu_layer {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  Matrix* _prev_parGrads;
  Matrix* _tmp_exp;
  Matrix* _tmp_sum;
  void _init(int64 input_s, int64 bs);
public:
  relu_layer();
  relu_layer(int64 input_s, int64 bs);
  void feed(Matrix* input);
  void feedGrad(Matrix* prev_parGrads);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
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
  void _init(int64 input_s, int64 bs);
public:
  sigmoid();
  sigmoid(int64 input_s, int64 bs);
  void feed(Matrix* input);
  void feedGrad(Matrix* prev_parGrads);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  Matrix* getBprop() const;
  ~sigmoid();
};


class conv2d_layer {
protected:
  int64 _batch;
  int64 _in_height;
  int64 _in_width;
  int64 _filter_height;
  int64 _filter_width;
  int64 _in_channels;
  int64 _out_channels;
  int64 _stride;
  float* weight_init;
  Matrix* _input;
  Matrix* _weight;
  Matrix* _output;
  cudaMatrix* _cudaInput;
  cudaMatrix* _cudaOutput;   // a pointer so that could be used for later layers
  cudaMatrix* _cudaWeight;
  bool _GPU;

  void _init(int64 batch, int64 in_height, int64 in_width, 
            int64 filter_height, int64 filter_width, int64 in_channels,
            int64 out_channels, int64 stride, bool GPU);
public:
  conv2d_layer();
  conv2d_layer(int64 batch, int64 in_height, int64 in_width, 
            int64 filter_height, int64 filter_width, int64 in_channels,
            int64 out_channels, int64 stride);
  conv2d_layer(int64 batch, int64 in_height, int64 in_width, 
            int64 filter_height, int64 filter_width, int64 in_channels,
            int64 out_channels, int64 stride, bool GPU);
  void loadW(const char* path);
  void feed(Matrix* input);
  void feed(cudaMatrix* input);
  void forward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  ~conv2d_layer();
};




class softmax_layer {
protected:
  Matrix* _input;
  Matrix* _output;   // a pointer so that could be used for later layers
  Matrix* _parGrads; // partial gradients
  Matrix * _tmp_exp;
  Matrix * _tmp_exp_sum;
  void _init(int64 input_s, int64 bs);
public:
  softmax_layer();
  softmax_layer(int64 input_s, int64 bs);
  void feed(Matrix* input);
  void forward(PASS_TYPE pass_type);
  void backward(PASS_TYPE pass_type);
  Matrix* getFprop() const;
  Matrix* getBprop() const;
  ~softmax_layer();
};

// class cross_entropy {
// protected:
//   Matrix* _logit;
//   Matrix* _label;
//   float* _output;   // a pointer so that could be used for later layers
//   Matrix* _parGrads; // partial gradients
//   void _init(int64 input_s);
// public:
//   cross_entropy();
//   cross_entropy(int64 input_s);
//   void feed(Matrix* logit, Matrix* label);
//   void forward(PASS_TYPE pass_type);
//   void backward(PASS_TYPE pass_type);
//   float* getFprop() const;
//   Matrix* getBprop() const;
//   ~cross_entropy();
// };


#endif

