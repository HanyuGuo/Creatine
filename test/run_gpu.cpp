#include "../include/layers/layer.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <cstring>
using namespace std;


void myprint(cudaMatrix * a) {
  float* data = new float[(a->getNumRows())* (a->getNumCols())];
  a -> getDeviceData(data);
  cout << "row: " << a->getNumRows()<< "col: " << a->getNumCols()<<endl;
  for (int i = 0; i < a->getNumRows(); i++) {
    for (int j = 0; j < a->getNumCols(); j++) {
      cout << data[i* (a->getNumCols()) + j] << " ";
    }
    cout << endl;
  }
}

int main(void) {
  bool GPU = true;


  float * test_x = new float[10000*784];
  float * test_y = new float[10000*10];
  load("../data/mnist_images.txt", test_x);
  load("../data/mnist_labels.txt", test_y);

  //create layers
  int hidden = 200;
  int bs = 20;
  float * x = new float[bs*784];
  float * y = new float[bs*10];
  int * argmax_label = new int[bs];
  int * argmax_pred = new int[bs];
  ip_layer * input_layer = new ip_layer(bs,784, GPU);
  ip_layer * label_layer = new ip_layer(bs,10, GPU);
  fc_layer * l1 = new fc_layer(bs,784,hidden,true, GPU);
  sigmoid * sig = new sigmoid(bs,hidden, GPU);
  fc_layer * l2 = new fc_layer(bs,hidden,10,true, GPU);
  softmax_layer * sfmx = new softmax_layer(bs,10, GPU);
  l1 -> loadW("../data/layer_1_weight.txt");
  l1 -> loadB("../data/layer_1_bias.txt");
  l2 -> loadW("../data/layer_2_weight.txt");
  l2 -> loadB("../data/layer_2_bias.txt");

  int correct = 0;
  for(int i=0; i<500; i++) {
    memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
    memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
    input_layer -> feed(x, bs*784);
    label_layer -> feed(y, bs*10);
    l1->feed(input_layer->getFprop(GPU));
    l1->forward(PASS_TRAIN);

    sig->feed(l1->getFprop(GPU));
    sig->forward(PASS_TRAIN);
    cudaMatrix * peek = l1->getFprop(GPU);
    // myprint(peek);
    l2->feed(sig->getFprop(GPU));
    l2->forward(PASS_TRAIN);
    peek = l2->getFprop(GPU);
    myprint(peek);

    // peek = label_layer->getFprop(GPU); 
    // myprint(peek);
    sfmx->feed(l2->getFprop(GPU));
    sfmx->forward(PASS_TRAIN);
    peek = sfmx->getFprop(GPU);
    myprint(peek);
    cudaMatrix* Matrix_sftmx = sfmx->getFprop(GPU);
    cudaMatrix* label = label_layer->getFprop(GPU);

    Matrix_sftmx -> argmax_gpu(argmax_pred);
    label -> argmax_gpu(argmax_label);
    correct += Equal(argmax_pred, argmax_label, bs);;
    cout << "current accurarcy: " << correct << " "<< float(correct) / float((i+1)*bs)<< endl; 

  }
  cout << "\nPass!" << endl;
  // delete [] argmax_label;
  // delete [] argmax_pred;
  delete [] x;
  delete [] y;
  delete [] test_x;
  delete [] test_y;
  delete input_layer;
  // delete label_layer;
  // delete l1;
  // delete l2;
  // delete sfmx;
  return 0;
}