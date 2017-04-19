#include "../include/layers/layer.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <cstring>
using namespace std;

int main(void) {
  bool GPU = true;
  float * x = new float[10*784];
  float * y = new float[10*10];

  float * test_x = new float[10000*784];
  float * test_y = new float[10000*10];
  load("../data/mnist_images.txt", test_x);
  load("../data/mnist_labels.txt", test_y);
  // float * argmax_label = new float[10];
  // float * argmax_pred = new float[10];
  //create layers
  int hidden = 200;
  int bs = 10;
  ip_layer * input_layer = new ip_layer(bs,784, GPU);
  ip_layer * label_layer = new ip_layer(bs,10, GPU);
  fc_layer * l1 = new fc_layer(bs,784,hidden,true, GPU);
  sigmoid * sig = new sigmoid(bs,hidden, GPU);
  // fc_layer * l2 = new fc_layer(hidden,10,10);
  // softmax_layer * sfmx = new softmax_layer(10,10);
  // l1 -> loadW("../data/layer_1_weight.txt");
  // l1 -> loadB("../data/layer_1_bias.txt");
  // l2 -> loadW("../data/layer_2_weight.txt");
  // l2 -> loadB("../data/layer_2_bias.txt");

  int correct = 0;
  for(int i=0; i<1; i++) {
    memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
    memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
    input_layer -> feed(x, bs*784);
    label_layer -> feed(y, bs*10);
    l1->feed(input_layer->getFprop(GPU));
    l1->forward(PASS_TRAIN);

    sig->feed(l1->getFprop());
    sig->forward(PASS_TRAIN);

    // l2->feed(sig->getFprop());
    // l2->forward(PASS_TRAIN);

    // sfmx->feed(l2->getFprop());
    // sfmx->forward(PASS_TRAIN);
    // Matrix* Matrix_sftmx = sfmx->getFprop();
    // Matrix* label = label_layer->getFprop();

    // Matrix_sftmx -> argmax(argmax_pred);
    // label -> argmax(argmax_label);
    // correct += Equal(argmax_pred, argmax_label, 10);;
    // cout << "current accurarcy: " << float(correct) / float(i*10)<< endl; 

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