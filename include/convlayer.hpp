#ifndef _CONVOLUTIONAL_LAYER_HPP_
#define _CONVOLUTIONAL_LAYER_HPP_
#include "Activation.cuh"

class ConvLayer {
private:
   std::string fromLayer;
   std::string toLayer;
   bool gradProducer; // True if this layer produces gradient.
   cudaMatrix *_act; // forward pass activation computation
   cudaMatrix *ldata; // Layer data.
  //  float dropout;
   int stride;
   int pad;
   int height,width;
   int size;

public:
  ConvLayer (float *data, int numrows,int numcols,int stride, int pad, Activation a); // create a convolutional layer with params from cfg files.
  virtual ~ConvLayer ();
  std::string getToLayer() const {return toLayer;}
  std::string getFromLayer() const {return fromLayer;}
  bool isgradProducer() const {return gradProducer;}
  cudaMatrix* getLayerData() const {return data;}
  cudaMatrix* getActivation() const {return _act;}
  void compute_fwdpass_gpu(cudaMatrix &c, Activation a, cudaMatrix & act);

};






#endif
