#ifndef LAYER_CU_H
#define LAYER_CU_H 
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <assert.h>
#include <gpmatrix.cuh>
using namespace std;


/* 
Abstract Layer Class: Base Class from which all other layers will inherit.
Contains mostly pure virtual functions for overriding by base classes. 

All derived classes MUST override the forward and backward pass functions.





*/

enum PASSTYPE{TRAIN,TEST,VAL};

typedef std::vector<GpMatrix*> GpMatrixVec;
class Layer
{
protected:
  std::vector<Layer *>_prev_layer,_nxt_layer;
  bool gradProducer; // check if layer produces a gradient
  std::string name,type;

  virtual void fPropActivations() = 0;
  virtual void bprop();
public:
	Layer();
	~Layer();
	virtual void fwd_pass_cpu(Matrix &mat, PASSTYPE type);
	virtual void fwd_pass_gpu(GpMatrix &gp_mat, PASSTYPE type);
	virtual void bwd_pass_cpu(Matrix &mat, PASSTYPE type);
	virtual void bwd_pass_gpu(GpMatrix &gp_mat, PASSTYPE type);

	std::string getName() {
		return name;
	}

	std::string getType() {
		return type;
	}

	virtual void reset();
	bool isGradientProducer();
	bool isGradientConsumer();
	void cpCPU(); // call the Matrix functions and use CPU. Layer stuff unclear.
	void cpGPU(); // call the GpMatrix functions and use GPU.


	
};

#endif
