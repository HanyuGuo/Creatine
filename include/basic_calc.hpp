#ifndef BASIC_CALC_H
#define BASIC_CALC_H

#include <stdlib.h>
#include <math.h>
#include <algorithm>


inline double _add(double x, double y) {
	return x + y;
}

inline double _addScaleY(double x, double y, double scale) {
	return x + y * scale;
}

inline double _divide(double x, double y) {
	return x / y;
}

inline double _exp(double x) {
	return exp(x);
}

#endif 