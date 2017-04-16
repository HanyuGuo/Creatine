#ifndef UTILS_H
#define	UTILS_H
#include "assert.h"
#include <fstream>
#include <string>
#include <iostream>


enum PASS_TYPE {PASS_TRAIN, PASS_TEST, PASS_GC};
void load(const char* path, double* weights);
int Equal(const double * a,const double * b, int len);
void argmax(const double* input, double* result);
#endif