#ifndef BASIC_CALC_H
#define BASIC_CALC_H

#include <stdlib.h>
#include <math.h>
#include <algorithm>


inline float _identity(float x) {
  return x;
}

inline float _add(float x, float y) {
  return x + y;
}


inline float _addScaleY(float x, float y, float scale) {
  return x + y * scale;
}

inline float _mul(float x, float y) {
	return x * y;
}

inline float _divide(float x, float y) {
  return x / y;
}

inline float _divided(float x, float y) {
  return y / x;
}


inline float _exp(float x) {
  return exp(x);
}

inline float _ln(float x) {
  return log(x + 1e-5);
}


inline float _log(float x) {
  return log2(x + 1e-5);
}

inline float _max(float x, float y) {
  return x>=y ? x : y;
}


#endif 