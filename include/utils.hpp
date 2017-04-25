#ifndef UTILS_H
#define	UTILS_H
#include "assert.h"
#include <fstream>
#include <string>
#include <iostream>


enum PASS_TYPE {PASS_TRAIN, PASS_TEST, PASS_GC};
void load(const char* path, float* weights);
int Equal(const int * a,const int * b, int len);
void argmax(const float* input, int* result);
#endif