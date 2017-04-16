#ifndef BASIC_CALC_H
#define BASIC_CALC_H

#include <stdlib.h>
#include <math.h>
#include <algorithm>


inline double _identity(double x) {
  return x;
}

inline double _add(double x, double y) {
  return x + y;
}


inline double _addScaleY(double x, double y, double scale) {
  return x + y * scale;
}

inline double _mul(double x, double y) {
	return x * y;
}

inline double _divide(double x, double y) {
  return x / y;
}

inline double _divided(double x, double y) {
  return y / x;
}


inline double _exp(double x) {
  return exp(x);
}

inline double _ln(double x) {
  return log(x + 1e-5);
}


inline double _log(double x) {
  return log2(x + 1e-5);
}

inline double _max(double x, double y) {
  return x>=y ? x : y;
}


#endif 