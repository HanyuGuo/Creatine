#include "../include/layers/layer.hpp"
#include <iostream>
#include <cstring>
using namespace std;

int main(void) {
  double * x = new double[2];
  double * y = new double[2];
  double tr_x[] = {0,1,1,2,3,5,5,3,2,1,1,3,3,4};
  double tr_y[] = {1,0,1,0,1,0,0,1,0,1,1,0,0,1};
  double te_x[] = {0,1,2,1};
  double te_y[] = {1,0,0,1};


  int epoch = 5;
  //create layers
  ip_layer * input_layer = new ip_layer(2);
  ip_layer * label_layer = new ip_layer(2);
  fc_layer * l1 = new fc_layer(2,10);
  relu_layer * relu = new relu_layer(10);
  fc_layer * l2 = new fc_layer(10,2);
  softmax_layer * sfmx = new softmax_layer(2);
  cross_entropy * cost  = new cross_entropy(2);

  for(int j = 0; j < epoch; j++){
    for(int i = 0; i < 7; i++){
      //forward
      memcpy(x, tr_x +i*2, 2*sizeof(double));
      memcpy(y, tr_y +i*2, 2*sizeof(double));
      // cout <<"x:"<< x[0] << " "<< x[1] << endl;
      // cout <<"y:"<< y[0] << " "<< y[1] << endl;
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

      // Matrix* Matrix_t = sfmx->getFprop();
      // cout << "relu layer" << endl;
      // for (int i=0; i<2; i++) 
      //   cout << Matrix_t-> getCell(0,i) <<" ";
      // cout << endl;

      cost -> feed(sfmx -> getFprop(), label_layer -> getFprop());
      cost -> forward(PASS_TRAIN);
      double* ce = cost->getFprop();
      cout << "cost: "<< *ce << endl;
      cost -> backward(PASS_TRAIN);

      //backward

      l2 -> feedGrad(cost -> getBprop());
      // Matrix* cg = cost->getBprop();
      // cout << cg->getCell(0,0) << " " << cg->getCell(0,1) << endl;
      l2 -> backward(PASS_TRAIN);  
      l2 -> calcParGrads(PASS_TRAIN);
      l2 -> applyGrads(0.1);

      // relu -> feedGrad(l2 -> getBprop());
      // relu -> backward(PASS_TRAIN);
      // Matrix * rr = relu -> getBprop();
      // cout << rr->getCell(0,0)<< " "<< rr->getCell(0,1)<< " "<<rr->getCell(0,2)<< " "<<rr->getCell(0,3)<< " "<<rr->getCell(0,4) << endl;
      l1 -> feedGrad(l2 -> getBprop());
      l1 -> backward(PASS_TRAIN);  
      l1 -> calcParGrads(PASS_TRAIN);
      l1 -> applyGrads(0.1);
    }
    cout <<"------------------------------------"<< endl;
  }
  for(int i=0; i<2; i++) {
    memcpy(x, te_x + i*2, 2*sizeof(double));
    memcpy(y, te_y + i*2, 2*sizeof(double));
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
  return 0;
}