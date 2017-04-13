#ifndef _CONVOLUTIONAL_LAYER_HPP_
#define _CONVOLUTIONAL_LAYER_HPP_

class ConvLayer {
private:
   std::string fromLayer;
   std::string toLayer;
   bool gradProducer; // True if this layer produces gradient.


public:
  ConvLayer (); // create a convolutional layer with params from cfg files.
  virtual ~ConvLayer ();
  std::string getToLayer() const {return toLayer;}
  std::string getFromLayer() const {return fromLayer;}
  bool isgradProducer() const {return gradProducer;}
};






#endif
