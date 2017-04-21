#ifndef UTILS_H
#define	UTILS_H
#include "assert.h"
#include <fstream>
#include <string>
#include <iostream>


enum PASS_TYPE {PASS_TRAIN, PASS_TEST, PASS_GC};
void load(const char* path, float* weights);
int Equal(const float * a,const float * b, int len);
void argmax(const float* input, float* result);
#endif