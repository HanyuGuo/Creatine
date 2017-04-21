#include "../include/layers/layer.hpp"
#include <iostream>
#include <cstring>
using namespace std;

int main(void) {
  float * x = new float[2];
  float * y = new float[2];
  float tr_x[] = {0,1,1,2,3,5,5,3,2,1,1,3,3,4};
  float tr_y[] = {1,0,1,0,1,0,0,1,0,1,1,0,0,1};
  float te_x[] = {0,1,2,1};
  float te_y[] = {1,0,0,1};


  int epoch = 100;
  //create layers
  int hidden = 20;
  ip_layer * input_layer = new ip_layer(2);
  ip_layer * label_layer = new ip_layer(2);
  fc_layer * l1 = new fc_layer(2,hidden);
  sigmoid * relu = new sigmoid(hidden);
  fc_layer * l2 = new fc_layer(hidden,2);
  softmax_layer * sfmx = new softmax_layer(2);
  cross_entropy * cost  = new cross_entropy(2);

  for(int j = 0; j < epoch; j++){
    for(int i = 0; i < 7; i++){
      //forward
      memcpy(x, tr_x +i*2, 2*sizeof(float));
      memcpy(y, tr_y +i*2, 2*sizeof(float));
      // cout <<"x:"<< x[0] << " "<< x[1] << endl;
      // cout <<"y:"<< y[0] << " "<< y[1] << endl;
      input_layer -> feed(x);
      label_layer -> feed(y);
      l1->feed(input_layer->getFprop());
      l1->forward(PASS_TRAIN);
      Matrix* l2g = l1->getFprop();
      cout << "l1 :";
      for (int i = 0; i < l2g -> getNumRows(); i++) {
        for (int j = 0; j < l2g -> getNumCols(); j++) {
          cout << l2g->getCell(i,j) << " " ;
        }
      }
      cout  << endl;

      // relu->feed(l1->getFprop());
      // relu->forward(PASS_TRAIN);
      // l2g = relu->getFprop();
      // cout << "sigmoid :";
      // for (int i = 0; i < l2g -> getNumRows(); i++) {
      //   for (int j = 0; j < l2g -> getNumCols(); j++) {
      //     cout << l2g->getCell(i,j) << " " ;
      //   }
      // }
      // cout  << endl;


      l2->feed(l1->getFprop());
      l2->forward(PASS_TRAIN);
      l2g = l2->getFprop();
      cout << "l2 :";
      for (int i = 0; i < l2g -> getNumRows(); i++) {
        for (int j = 0; j < l2g -> getNumCols(); j++) {
          cout << l2g->getCell(i,j) << " " ;
        }
      }
      cout  << endl;
      // cout << l2g->getCell(0,2) << " " << l2g->getCell(0,3) << " ";
      // cout << l2g->getCell(0,4) << endl;
      sfmx->feed(l2->getFprop());
      sfmx->forward(PASS_TRAIN);

      l2g = sfmx->getFprop();
      cout << "softmax_layer :";
      for (int i = 0; i < l2g -> getNumRows(); i++) {
        for (int j = 0; j < l2g -> getNumCols(); j++) {
          cout << l2g->getCell(i,j) << " " ;
        }
      }
      cout  << endl;

      cost -> feed(sfmx -> getFprop(), label_layer -> getFprop());
      cost -> forward(PASS_TRAIN);
      float* ce = cost->getFprop();
      cout << "cost: "<< *ce << endl;
      cost -> backward(PASS_TRAIN);

      //backward

      l2 -> feedGrad(cost -> getBprop());
      Matrix* cg = cost->getBprop();
      cout << "cost bp :";
      cout << cg->getCell(0,0) << " " << cg->getCell(0,1) << endl;
      l2 -> backward(PASS_TRAIN);  
      l2 -> calcParGrads(PASS_TRAIN);
      l2 -> applyGrads(0.1);

      l2g = l2->getBprop();
      cout << "l2 bp :";
      for (int i = 0; i < l2g -> getNumRows(); i++) {
        for (int j = 0; j < l2g -> getNumCols(); j++) {
          cout << l2g->getCell(i,j) << " " ;
        }
      }
      cout  << endl;

      // relu -> feedGrad(l2 -> getBprop());
      // relu -> backward(PASS_TRAIN);
      // cout << "relu bp :";
      // l2g = relu->getBprop();
      // for (int i = 0; i < l2g -> getNumRows(); i++) {
      //   for (int j = 0; j < l2g -> getNumCols(); j++) {
      //     cout << l2g->getCell(i,j) << " " ;
      //   }
      // }
      // cout  << endl;

      l1 -> feedGrad(l2-> getBprop());
      // l1 -> backward(PASS_TRAIN);  
      l1 -> calcParGrads(PASS_TRAIN);
      l1 -> applyGrads(0.1);
      cout <<"------------------------------------"<< endl << endl;
    }
    cout <<"------------------------------------"<< endl << endl;
  }
  for(int i=0; i<2; i++) {
    memcpy(x, te_x + i*2, 2*sizeof(float));
    memcpy(y, te_y + i*2, 2*sizeof(float));
    input_layer -> feed(x);
    label_layer -> feed(y);
    l1->feed(input_layer->getFprop());
    l1->forward(PASS_TRAIN);

    // relu->feed(l1->getFprop());
    // relu->forward(PASS_TRAIN);

    l2->feed(l1->getFprop());
    l2->forward(PASS_TRAIN);

    sfmx->feed(l2->getFprop());
    sfmx->forward(PASS_TRAIN);
    Matrix* Matrix_sftmx = sfmx->getFprop();
    for (int i=0; i<2; i++) 
      cout << Matrix_sftmx-> getCell(0,i)<<" ";

  }
  cout << "\nPass!" << endl;
  delete [] x;
  delete [] y;
  delete input_layer;
  delete label_layer;
  delete l1;
  delete l2;
  delete sfmx;
  delete cost;
  return 0;
}