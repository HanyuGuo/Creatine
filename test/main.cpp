#include "../include/layers/layer.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <cstring>
#include <ctime>
using namespace std;




int main(int argc, char *argv[]) {
  if (argc == 2) {
    if(strcmp(argv[1],"1")==0) {  
      printf("                Option 1 -- Fully Connected Layer TOY Example              \n");
      printf("Input:              MNIST digits              [batch, input_size]     =    [5,784]        \n");
      printf("Hidden Layer:       Fully Connecte Layer      [input_size, hidden]    =    [784,50]      \n");
      printf("Activation Layer:   RELU         \n");
      printf("Output Layer:       Fully Connecte Layer      [hidden, output_size]   =    [50,10]       \n");
      bool GPU = true;
      float * test_x = new float[10000*784];
      float * test_y = new float[10000*10];
      load("./data/mnist_images.txt", test_x);
      load("./data/mnist_labels.txt", test_y);
      //create layers
      int hidden = 50;
      int bs = 5;
      float * x = new float[bs*784];
      float * y = new float[bs*10];
      int * argmax_label = new int[bs];
      int * argmax_pred = new int[bs];
      ip_layer * input_layer = new ip_layer(bs,784, GPU);
      ip_layer * label_layer = new ip_layer(bs,10, GPU);
      fc_layer * l1 = new fc_layer(bs,784,hidden,true, GPU);
      relu_layer * relu = new relu_layer(bs,hidden, GPU);
      fc_layer * l2 = new fc_layer(bs,hidden,10,true, GPU);
      softmax_layer * sfmx = new softmax_layer(bs,10, GPU);
      l1 -> loadW("./data/fc_toy/t_h1.txt");
      l1 -> loadB("./data/fc_toy/t_b1.txt");
      l2 -> loadW("./data/fc_toy/t_hout.txt");
      l2 -> loadB("./data/fc_toy/t_bout.txt");

      int correct = 0;
      clock_t t1 = clock();
      for(int i=0; i<2000; i++) {
        memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
        memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
        input_layer -> feed(x, bs*784);
        label_layer -> feed(y, bs*10);
        l1->feed(input_layer->getFprop(GPU));
        l1->forward(PASS_TRAIN);

        relu->feed(l1->getFprop(GPU));
        relu->forward(PASS_TRAIN);

        l2->feed(relu->getFprop(GPU));
        l2->forward(PASS_TRAIN);


        sfmx->feed(l2->getFprop(GPU));
        sfmx->forward(PASS_TRAIN);


        cudaMatrix* Matrix_sftmx = sfmx->getFprop(GPU);
        cudaMatrix* label = label_layer->getFprop(GPU);

        Matrix_sftmx -> argmax_gpu(argmax_pred);
        label -> argmax_gpu(argmax_label);
        correct += Equal(argmax_pred, argmax_label, bs);;
        if (i % 5 == 0)
          cout << "Step : " << i << " , Current accurarcy: " << float(correct) / float((i+1)*bs)<< endl; 
      }
      cout << "Step : " << 2000 << " , Current accurarcy: " << float(correct) / float(10000)<< endl; 
      clock_t t2 = clock();
      std::cout << "Fully Connected Layer forward pass toy example spends "
                << double(t2 - t1) / CLOCKS_PER_SEC
                << " seconds\n";

    }
    else if (strcmp(argv[1],"2")==0) {
      printf("                Option 2 -- Fully Connected Layer WIDE Example              \n");
      printf("Input:                MNIST digits             [batch, input_size]       =    [20,784]        \n");
      printf("Hidden Layer 1:       Fully Connecte Layer     [input_size, hidden1]     =    [784,2000]      \n");
      printf("Activation Layer 1:   RELU         \n");
      printf("Hidden Layer 2:       Fully Connecte Layer     [hidden1, hidden2]        =    [2000,4000]      \n");
      printf("Activation Layer 2:   RELU         \n");
      printf("Hidden Layer 3:       Fully Connecte Layer     [hidden2, hidden3]        =    [4000,2000]      \n");
      printf("Activation Layer 3:   RELU         \n");
      printf("Output Layer:         Fully Connecte Layer     [hidden3, output_size]    =    [2000,10]       \n");
      bool GPU = true;
      float * test_x = new float[10000*784];
      float * test_y = new float[10000*10];
      load("./data/mnist_images.txt", test_x);
      load("./data/mnist_labels.txt", test_y);
      //create layers
      int bs = 20;
      int hidden1 = 2000;
      int hidden2 = 4000;
      int hidden3 = 2000;
      float * x = new float[bs*784];
      float * y = new float[bs*10];
      int * argmax_label = new int[bs];
      int * argmax_pred = new int[bs];
      ip_layer * input_layer = new ip_layer(bs,784, GPU);
      ip_layer * label_layer = new ip_layer(bs,10, GPU);
      fc_layer * l1 = new fc_layer(bs,784,hidden1,true, GPU);
      relu_layer * relu1 = new relu_layer(bs,hidden1, GPU);
      fc_layer * l2 = new fc_layer(bs,hidden1,hidden2,true, GPU);
      relu_layer * relu2 = new relu_layer(bs,hidden2, GPU);
      fc_layer * l3 = new fc_layer(bs,hidden2,hidden3,true, GPU);
      relu_layer * relu3 = new relu_layer(bs,hidden3, GPU);
      fc_layer * out = new fc_layer(bs,hidden3,10,true, GPU);
      softmax_layer * sfmx = new softmax_layer(bs,10, GPU);
      l1 -> loadW("./data/fc_wide/w_h1.txt");
      l1 -> loadB("./data/fc_wide/w_b1.txt");
      l2 -> loadW("./data/fc_wide/w_h2.txt");
      l2 -> loadB("./data/fc_wide/w_b2.txt");
      l3 -> loadW("./data/fc_wide/w_h3.txt");
      l3 -> loadB("./data/fc_wide/w_b3.txt");
      out -> loadW("./data/fc_wide/w_hout.txt");
      out -> loadB("./data/fc_wide/w_bout.txt");

      int correct = 0;
      clock_t t1 = clock();
      for(int i=0; i<10000/bs; i++) {
        memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
        memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
        input_layer -> feed(x, bs*784);
        label_layer -> feed(y, bs*10);
        l1->feed(input_layer->getFprop(GPU));
        l1->forward(PASS_TRAIN);
        relu1->feed(l1->getFprop(GPU));
        relu1->forward(PASS_TRAIN);

        l2->feed(relu1->getFprop(GPU));
        l2->forward(PASS_TRAIN);
        relu2->feed(l2->getFprop(GPU));
        relu2->forward(PASS_TRAIN);

        l3->feed(relu2->getFprop(GPU));
        l3->forward(PASS_TRAIN);
        relu3->feed(l3->getFprop(GPU));
        relu3->forward(PASS_TRAIN);

        out->feed(relu3->getFprop(GPU));
        out->forward(PASS_TRAIN);

        sfmx->feed(out->getFprop(GPU));
        sfmx->forward(PASS_TRAIN);


        cudaMatrix* Matrix_sftmx = sfmx->getFprop(GPU);
        cudaMatrix* label = label_layer->getFprop(GPU);

        Matrix_sftmx -> argmax_gpu(argmax_pred);
        label -> argmax_gpu(argmax_label);
        correct += Equal(argmax_pred, argmax_label, bs);;
        if (i % 5 == 0)
          cout << "Step : " << i << " , Current accurarcy: " << float(correct) / float((i+1)*bs)<< endl; 
      }
      cout << "Step : " << 10000/bs << " , Current accurarcy: " << float(correct) / float(10000)<< endl; 
      clock_t t2 = clock();
      std::cout << "Fully Connected Layer forward pass wide example spends "
                << double(t2 - t1) / CLOCKS_PER_SEC
                << " seconds\n";

    }
    else if (strcmp(argv[1],"3")==0) {
      printf("                Option 3 -- Fully Connected Layer DEEP Example              \n");
      printf("Input:                MNIST digits             [batch, input_size]       =    [20,784]        \n");
      printf("Hidden Layer 1:       Fully Connecte Layer     [input_size, hidden1]     =    [784,256]      \n");
      printf("Activation Layer 1:   RELU         \n");
      printf("Hidden Layer 2:       Fully Connecte Layer     [hidden1, hidden2]        =    [256,512]      \n");
      printf("Activation Layer 2:   RELU         \n");
      printf("Hidden Layer 3:       Fully Connecte Layer     [hidden2, hidden3]        =    [512,512]      \n");
      printf("Activation Layer 3:   RELU         \n");
      printf("Hidden Layer 4:       Fully Connecte Layer     [hidden3, hidden4]        =    [512,1024]      \n");
      printf("Activation Layer 4:   RELU         \n");   
      printf("Hidden Layer 5:       Fully Connecte Layer     [hidden4, hidden5]        =    [1024,1024]      \n");
      printf("Activation Layer 5:   RELU         \n");   
      printf("Hidden Layer 6:       Fully Connecte Layer     [hidden5, hidden6]        =    [1024,512]      \n");
      printf("Activation Layer 6:   RELU         \n");   
      printf("Hidden Layer 7:       Fully Connecte Layer     [hidden6, hidden7]        =    [512,512]      \n");
      printf("Activation Layer 7:   RELU         \n");   
      printf("Hidden Layer 8:       Fully Connecte Layer     [hidden7, hidden8]        =    [512,256]      \n");
      printf("Activation Layer 8:   RELU         \n");
      printf("Output Layer:         Fully Connecte Layer     [hidden8, output_size]    =    [256,10]       \n");

      bool GPU = true;
      float * test_x = new float[10000*784];
      float * test_y = new float[10000*10];
      load("./data/mnist_images.txt", test_x);
      load("./data/mnist_labels.txt", test_y);
      //create layers
      int bs = 20;
      int hidden1 = 256;
      int hidden2 = 512;
      int hidden3 = 512;
      int hidden4 = 1024;
      int hidden5 = 1024;
      int hidden6 = 512;
      int hidden7 = 512;
      int hidden8 = 256;
      float * x = new float[bs*784];
      float * y = new float[bs*10];
      int * argmax_label = new int[bs];
      int * argmax_pred = new int[bs];
      ip_layer * input_layer = new ip_layer(bs,784, GPU);
      ip_layer * label_layer = new ip_layer(bs,10, GPU);
      fc_layer *   l1 = new fc_layer(bs,784,hidden1,true, GPU);
      relu_layer * relu1 = new relu_layer(bs,hidden1, GPU);
      fc_layer *   l2 = new fc_layer(bs,hidden1,hidden2,true, GPU);
      relu_layer * relu2 = new relu_layer(bs,hidden2, GPU);
      fc_layer *   l3 = new fc_layer(bs,hidden2,hidden3,true, GPU);
      relu_layer * relu3 = new relu_layer(bs,hidden3, GPU);
      fc_layer *   l4 = new fc_layer(bs,hidden3,hidden4,true, GPU);
      relu_layer * relu4 = new relu_layer(bs,hidden4, GPU);
      fc_layer *   l5 = new fc_layer(bs,hidden4,hidden5,true, GPU);
      relu_layer * relu5 = new relu_layer(bs,hidden5, GPU);
      fc_layer *   l6 = new fc_layer(bs,hidden5,hidden6,true, GPU);
      relu_layer * relu6 = new relu_layer(bs,hidden6, GPU);
      fc_layer *   l7 = new fc_layer(bs,hidden6,hidden7,true, GPU);
      relu_layer * relu7 = new relu_layer(bs,hidden7, GPU);
      fc_layer *   l8 = new fc_layer(bs,hidden7,hidden8,true, GPU);
      relu_layer * relu8 = new relu_layer(bs,hidden8, GPU);
      fc_layer *   out = new fc_layer(bs,hidden8,10,true, GPU);
      softmax_layer * sfmx = new softmax_layer(bs,10, GPU);
      l1  -> loadW("./data/fc_deep/d_h1.txt");
      l1  -> loadB("./data/fc_deep/d_b1.txt");
      l2  -> loadW("./data/fc_deep/d_h2.txt");
      l2  -> loadB("./data/fc_deep/d_b2.txt");
      l3  -> loadW("./data/fc_deep/d_h3.txt");
      l3  -> loadB("./data/fc_deep/d_b3.txt");
      l4  -> loadW("./data/fc_deep/d_h4.txt");
      l4  -> loadB("./data/fc_deep/d_b4.txt");
      l5  -> loadW("./data/fc_deep/d_h5.txt");
      l5  -> loadB("./data/fc_deep/d_b5.txt");
      l6  -> loadW("./data/fc_deep/d_h6.txt");
      l6  -> loadB("./data/fc_deep/d_b6.txt");
      l7  -> loadW("./data/fc_deep/d_h7.txt");
      l7  -> loadB("./data/fc_deep/d_b7.txt");
      l8  -> loadW("./data/fc_deep/d_h8.txt");
      l8  -> loadB("./data/fc_deep/d_b8.txt");
      out -> loadW("./data/fc_deep/d_hout.txt");
      out -> loadB("./data/fc_deep/d_bout.txt");

      int correct = 0;
      clock_t t1 = clock();
      for(int i=0; i<10000/bs; i++) {
        memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
        memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
        input_layer -> feed(x, bs*784);
        label_layer -> feed(y, bs*10);

        l1->feed(input_layer->getFprop(GPU));
        l1->forward(PASS_TRAIN);
        relu1->feed(l1->getFprop(GPU));
        relu1->forward(PASS_TRAIN);

        l2->feed(relu1->getFprop(GPU));
        l2->forward(PASS_TRAIN);
        relu2->feed(l2->getFprop(GPU));
        relu2->forward(PASS_TRAIN);

        l3->feed(relu2->getFprop(GPU));
        l3->forward(PASS_TRAIN);
        relu3->feed(l3->getFprop(GPU));
        relu3->forward(PASS_TRAIN);

        l4->feed(relu3->getFprop(GPU));
        l4->forward(PASS_TRAIN);
        relu4->feed(l4->getFprop(GPU));
        relu4->forward(PASS_TRAIN);

        l5->feed(relu4->getFprop(GPU));
        l5->forward(PASS_TRAIN);
        relu5->feed(l5->getFprop(GPU));
        relu5->forward(PASS_TRAIN);

        l6->feed(relu5->getFprop(GPU));
        l6->forward(PASS_TRAIN);
        relu6->feed(l6->getFprop(GPU));
        relu6->forward(PASS_TRAIN);

        l7->feed(relu6->getFprop(GPU));
        l7->forward(PASS_TRAIN);
        relu7->feed(l7->getFprop(GPU));
        relu7->forward(PASS_TRAIN);

        l8->feed(relu7->getFprop(GPU));
        l8->forward(PASS_TRAIN);
        relu8->feed(l8->getFprop(GPU));
        relu8->forward(PASS_TRAIN);

        out->feed(relu8->getFprop(GPU));
        out->forward(PASS_TRAIN);

        sfmx->feed(out->getFprop(GPU));
        sfmx->forward(PASS_TRAIN);


        cudaMatrix* Matrix_sftmx = sfmx->getFprop(GPU);
        cudaMatrix* label = label_layer->getFprop(GPU);

        Matrix_sftmx -> argmax_gpu(argmax_pred);
        label -> argmax_gpu(argmax_label);
        correct += Equal(argmax_pred, argmax_label, bs);;
        if (i % 5 == 0)
          cout << "Step : " << i << " , Current accurarcy: " << float(correct) / float((i+1)*bs)<< endl; 
      }
      cout << "Step : " << 10000/bs << " , Current accurarcy: " << float(correct) / float(10000)<< endl; 
      clock_t t2 = clock();
      std::cout << "Fully Connected Layer forward pass deep example spends "
                << double(t2 - t1) / CLOCKS_PER_SEC
                << " seconds\n";
    }
else if (strcmp(argv[1],"4")==0) {
      printf("                Option 4 -- Fully Connected Layer DEEP and WIDE Example              \n");
      printf("Input:                MNIST digits             [batch, input_size]       =    [20,784]        \n");
      printf("Hidden Layer 1:       Fully Connecte Layer     [input_size, hidden1]     =    [784,2000]      \n");
      printf("Activation Layer 1:   RELU         \n");
      printf("Hidden Layer 2:       Fully Connecte Layer     [hidden1, hidden2]        =    [2000,2500]      \n");
      printf("Activation Layer 2:   RELU         \n");
      printf("Hidden Layer 3:       Fully Connecte Layer     [hidden2, hidden3]        =    [2500,3000]      \n");
      printf("Activation Layer 3:   RELU         \n");
      printf("Hidden Layer 4:       Fully Connecte Layer     [hidden3, hidden4]        =    [3000,4000]      \n");
      printf("Activation Layer 4:   RELU         \n");   
      printf("Hidden Layer 5:       Fully Connecte Layer     [hidden4, hidden5]        =    [4000,4000]      \n");
      printf("Activation Layer 5:   RELU         \n");   
      printf("Hidden Layer 6:       Fully Connecte Layer     [hidden5, hidden6]        =    [4000,3000]      \n");
      printf("Activation Layer 6:   RELU         \n");   
      printf("Hidden Layer 7:       Fully Connecte Layer     [hidden6, hidden7]        =    [3000,2500]      \n");
      printf("Activation Layer 7:   RELU         \n");   
      printf("Hidden Layer 8:       Fully Connecte Layer     [hidden7, hidden8]        =    [2500,2000]      \n");
      printf("Activation Layer 8:   RELU         \n");
      printf("Output Layer:         Fully Connecte Layer     [hidden8, output_size]    =    [2000,10]       \n");

      bool GPU = true;
      float * test_x = new float[10000*784];
      float * test_y = new float[10000*10];
      load("./data/mnist_images.txt", test_x);
      load("./data/mnist_labels.txt", test_y);
      //create layers
      int bs = 20;
      int hidden1 = 2000;
      int hidden2 = 2500;
      int hidden3 = 3000;
      int hidden4 = 4000;
      int hidden5 = 4000;
      int hidden6 = 3000;
      int hidden7 = 2500;
      int hidden8 = 2000;
      float * x = new float[bs*784];
      float * y = new float[bs*10];
      int * argmax_label = new int[bs];
      int * argmax_pred = new int[bs];
      ip_layer * input_layer = new ip_layer(bs,784, GPU);
      ip_layer * label_layer = new ip_layer(bs,10, GPU);
      fc_layer *   l1 = new fc_layer(bs,784,hidden1,true, GPU);
      relu_layer * relu1 = new relu_layer(bs,hidden1, GPU);
      fc_layer *   l2 = new fc_layer(bs,hidden1,hidden2,true, GPU);
      relu_layer * relu2 = new relu_layer(bs,hidden2, GPU);
      fc_layer *   l3 = new fc_layer(bs,hidden2,hidden3,true, GPU);
      relu_layer * relu3 = new relu_layer(bs,hidden3, GPU);
      fc_layer *   l4 = new fc_layer(bs,hidden3,hidden4,true, GPU);
      relu_layer * relu4 = new relu_layer(bs,hidden4, GPU);
      fc_layer *   l5 = new fc_layer(bs,hidden4,hidden5,true, GPU);
      relu_layer * relu5 = new relu_layer(bs,hidden5, GPU);
      fc_layer *   l6 = new fc_layer(bs,hidden5,hidden6,true, GPU);
      relu_layer * relu6 = new relu_layer(bs,hidden6, GPU);
      fc_layer *   l7 = new fc_layer(bs,hidden6,hidden7,true, GPU);
      relu_layer * relu7 = new relu_layer(bs,hidden7, GPU);
      fc_layer *   l8 = new fc_layer(bs,hidden7,hidden8,true, GPU);
      relu_layer * relu8 = new relu_layer(bs,hidden8, GPU);
      fc_layer *   out = new fc_layer(bs,hidden8,10,true, GPU);
      softmax_layer * sfmx = new softmax_layer(bs,10, GPU);
      l1  -> loadW("./data/fc_final/f_h1.txt");
      l1  -> loadB("./data/fc_final/f_b1.txt");
      l2  -> loadW("./data/fc_final/f_h2.txt");
      l2  -> loadB("./data/fc_final/f_b2.txt");
      l3  -> loadW("./data/fc_final/f_h3.txt");
      l3  -> loadB("./data/fc_final/f_b3.txt");
      l4  -> loadW("./data/fc_final/f_h4.txt");
      l4  -> loadB("./data/fc_final/f_b4.txt");
      l5  -> loadW("./data/fc_final/f_h5.txt");
      l5  -> loadB("./data/fc_final/f_b5.txt");
      l6  -> loadW("./data/fc_final/f_h6.txt");
      l6  -> loadB("./data/fc_final/f_b6.txt");
      l7  -> loadW("./data/fc_final/f_h7.txt");
      l7  -> loadB("./data/fc_final/f_b7.txt");
      l8  -> loadW("./data/fc_final/f_h8.txt");
      l8  -> loadB("./data/fc_final/f_b8.txt");
      out -> loadW("./data/fc_final/f_hout.txt");
      out -> loadB("./data/fc_final/f_bout.txt");

      int correct = 0;
      clock_t t1 = clock();
      for(int i=0; i<10000/bs; i++) {
        memcpy(x, test_x + bs*i*784, bs*784*sizeof(float));
        memcpy(y, test_y + bs*i*10, bs*10*sizeof(float));
        input_layer -> feed(x, bs*784);
        label_layer -> feed(y, bs*10);

        l1->feed(input_layer->getFprop(GPU));
        l1->forward(PASS_TRAIN);
        relu1->feed(l1->getFprop(GPU));
        relu1->forward(PASS_TRAIN);

        l2->feed(relu1->getFprop(GPU));
        l2->forward(PASS_TRAIN);
        relu2->feed(l2->getFprop(GPU));
        relu2->forward(PASS_TRAIN);

        l3->feed(relu2->getFprop(GPU));
        l3->forward(PASS_TRAIN);
        relu3->feed(l3->getFprop(GPU));
        relu3->forward(PASS_TRAIN);

        l4->feed(relu3->getFprop(GPU));
        l4->forward(PASS_TRAIN);
        relu4->feed(l4->getFprop(GPU));
        relu4->forward(PASS_TRAIN);

        l5->feed(relu4->getFprop(GPU));
        l5->forward(PASS_TRAIN);
        relu5->feed(l5->getFprop(GPU));
        relu5->forward(PASS_TRAIN);

        l6->feed(relu5->getFprop(GPU));
        l6->forward(PASS_TRAIN);
        relu6->feed(l6->getFprop(GPU));
        relu6->forward(PASS_TRAIN);

        l7->feed(relu6->getFprop(GPU));
        l7->forward(PASS_TRAIN);
        relu7->feed(l7->getFprop(GPU));
        relu7->forward(PASS_TRAIN);

        l8->feed(relu7->getFprop(GPU));
        l8->forward(PASS_TRAIN);
        relu8->feed(l8->getFprop(GPU));
        relu8->forward(PASS_TRAIN);

        out->feed(relu8->getFprop(GPU));
        out->forward(PASS_TRAIN);

        sfmx->feed(out->getFprop(GPU));
        sfmx->forward(PASS_TRAIN);


        cudaMatrix* Matrix_sftmx = sfmx->getFprop(GPU);
        cudaMatrix* label = label_layer->getFprop(GPU);

        Matrix_sftmx -> argmax_gpu(argmax_pred);
        label -> argmax_gpu(argmax_label);
        correct += Equal(argmax_pred, argmax_label, bs);;
        if (i % 5 == 0)
          cout << "Step : " << i << " , Current accurarcy: " << float(correct) / float((i+1)*bs)<< endl; 
      }
      cout << "Step : " << 10000/bs << " , Current accurarcy: " << float(correct) / float(10000)<< endl; 
      clock_t t2 = clock();
      std::cout << "Fully Connected Layer forward pass deep and wide example spends "
                << double(t2 - t1) / CLOCKS_PER_SEC
                << " seconds\n";
    }
  }
  else {
    printf("argc = %i\n", argc);
    printf("Creatine: A Simple Deep Learning Framework     - by Hanyu Guo, Ankit Kulshrestha \n");
    printf("Please uncompress data and place data folder under Creatine main folder.\n");
    printf("1. Fully Connected Layer TOY Example \n");
    printf("2. Fully Connected Layer WIDE Example \n");
    printf("3. Fully Connected Layer DEEP Example  \n");
    printf("4. Fully Connected Layer DEEP and WIDE Example  \n");
    return 0;
  }
}