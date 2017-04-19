#include "../include/utils.hpp"
#include "../include/matrix.hpp"


void load(const char* path, float* weights) {
  using namespace std;
  ifstream myfile;
  string infile;
  string line;
  string outfile;
  myfile.open(path);
  int i = 0;
  while (getline (myfile, line, ',')){
  	weights[i] = stod(line);
    i++;
  }
}


int Equal(const float * a,const float * b, int len) {
  int correct = 0;
  for (int j = 0; j < len; j++) {
    if(a[j] == b[j])
      correct ++;
  }
  return correct;
}

void argmax(const float* input, float* result, int w, int h) {
  for (int i=0; i < h; i++) {
    float temp = input[i*w];
    result[i] = 0;
    for (int j=1; j < w; j++) {
      if (temp < input[i*w+j]) {
        temp = input[i*w+j];
        result[i] = j;
      }
    }
  }
}
