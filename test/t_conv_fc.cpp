#include "../include/layers/layer.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <cstring>
using namespace std;


void myprint(string name, Matrix* m) {
  cout << "\n" << name << ": ";
  for (int i = 0; i < m -> getNumRows(); i++) {
    for (int j = 0; j < m -> getNumCols(); j++) {
      cout << m->getCell(i,j) << " " ;
    }
  }
  cout << "\n";
}

void myprint(string name, convMatrix* m) {
  cout << "\n" << name << ": ";
  for (int i = 0; i < m -> getDim(0); i++) {
    for (int j = 0; j < m -> getDim(1); j++) {
      for (int k = 0; k < m -> getDim(2); k++) {
        for (int l = 0; l < m -> getDim(3); l++) {
          cout << m->getCell(i,j,k,l) << " ";
        }
      }
    }
  }
  cout << "\n";
}


int main(void) {
  float * x = new float[10*784];
  float * y = new float[10*10];

  float * test_x = new float[10000*784];
  float * test_y = new float[10000*10];
  load("../data/mnist_images.txt", test_x);
  load("../data/mnist_labels.txt", test_y);

  //create layers
  int hidden = 1024;
  int bs = 1;
  float * argmax_label = new float[2];
  float * argmax_pred = new float[2];
  ip_layer * input_layer = new ip_layer(bs,28,28,1);
  ip_layer * label_layer = new ip_layer(bs,10);
  conv2d_layer * l1 = new conv2d_layer(bs,28,28,5,5,1,2,2);
  relu_layer * l2 = new relu_layer(bs,14,14,2);
  conv2d_layer * l3 = new conv2d_layer(bs,14,14,5,5,2,4,2);
  relu_layer * l4 = new relu_layer(bs,7,7,4);
  fc_layer * l5 = new fc_layer(bs, 7*7*4, hidden);
  relu_layer * l6 = new relu_layer(bs, hidden);
  fc_layer * l7 = new fc_layer(bs, hidden, 10);
  softmax_layer * l8 = new softmax_layer(bs,10);
  Matrix* pMatrix;
  convMatrix * convpMatrix;
  l1 -> loadW("../data/wc1.txt");
  l3 -> loadW("../data/wc2.txt");
  l5 -> loadW("../data/wd1.txt");
  l5 -> loadB("../data/bd1.txt");
  l7 -> loadW("../data/wout.txt");
  l7 -> loadB("../data/bout.txt");
  int correct = 0;
  for(int i=0; i<10; i++) {
    memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
    memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
    input_layer -> feed(x, bs*784);
    label_layer -> feed(y, bs*10);
    l1->feed(input_layer->getConvFprop());
    l1->forward(PASS_TRAIN);
    // myprint("l1", l1->getFprop());

    l2->feed(l1->getFprop());
    l2->forward(PASS_TRAIN);
    // myprint("l2", l2->getConvFprop());

    l3->feed(l2->getConvFprop());
    l3->forward(PASS_TRAIN);
    // myprint("l3", l3->getFprop());

    l4->feed(l3->getFprop());
    l4->forward(PASS_TRAIN);
    // myprint("l4", l4->getFlattenFprop());

    l5->feed(l4->getFlattenFprop());
    l5->forward(PASS_TRAIN);
    pMatrix = l5->getFprop();
    // cout << "\nl5 :";
    // for (int i = 0; i < pMatrix -> getNumRows(); i++) {
    //   for (int j = 0; j < pMatrix -> getNumCols(); j++) {
    //     cout << pMatrix->getCell(i,j) << " " ;
    //   }
    // }
    // cout << "\n";
    l6->feed(l5->getFprop());
    l6->forward(PASS_TRAIN);
    pMatrix = l6->getFprop();
    // cout << "\nl6 :";
    // for (int i = 0; i < pMatrix -> getNumRows(); i++) {
    //   for (int j = 0; j < pMatrix -> getNumCols(); j++) {
    //     cout << pMatrix->getCell(i,j) << " " ;
    //   }
    // }
    // cout << "\n";
    l7->feed(l6->getFprop());
    l7->forward(PASS_TRAIN);
    pMatrix = l7->getFprop();
    // cout << "\nl7 :";
    // for (int i = 0; i < pMatrix -> getNumRows(); i++) {
    //   for (int j = 0; j < pMatrix -> getNumCols(); j++) {
    //     cout << pMatrix->getCell(i,j) << " " ;
    //   }
    // }
    // cout << "\n";
    l8->feed(l7->getFprop());
    l8->forward(PASS_TRAIN);
    pMatrix = l8->getFprop();
    // cout << "\nsoftmax_layer :";
    // for (int i = 0; i < pMatrix -> getNumRows(); i++) {
    //   for (int j = 0; j < pMatrix -> getNumCols(); j++) {
    //     cout << pMatrix->getCell(i,j) << " " ;
    //   }
    // }

    cout  << endl;
    Matrix* Matrix_sftmx = l8->getFprop();
    Matrix* label = label_layer->getFprop();
    // myprint("label", label_layer->getFprop());

    Matrix_sftmx -> argmax(argmax_pred);
    label -> argmax(argmax_label);
    correct += Equal(argmax_pred, argmax_label, bs);;
    cout << "current accurarcy: " << float(correct) / float((i+1)*bs)<< endl; 

  }
  cout << "\nPass!" << endl;


  delete [] test_x;
  delete [] test_y;
  delete input_layer;
  delete label_layer;
  delete l1;
  delete l2;
  delete l3;
  delete l4;
  delete l5;
  delete l6;
  delete l7;
  delete l8;
  // delete [] x;
  // delete [] y;
  // delete [] argmax_label;
  // delete [] argmax_pred;
  // delete sfmx;
  return 0;
}