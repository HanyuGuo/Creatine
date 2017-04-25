#include "../include/convMatrix.hpp"
#include "../include/matrix.hpp"
#include <iostream>

using namespace std;

int main(void){
	float input_data[] = {1,1,2,0,1,
					      2,0,1,1,2,
					      0,2,2,1,0,
					      0,0,1,0,1,
					      0,2,0,0,1,
					      0,1,2,0,0,
					      2,2,1,0,2,
					      2,1,2,2,0,
					      1,0,1,0,1,
					      0,0,1,0,1,
					      1,0,1,2,2,
					      0,1,0,1,0,
					      0,2,2,1,1,
					      2,1,1,2,0,
					      2,1,0,1,2,
					  	  1,1,2,0,1,
					      2,0,1,1,2,
					      0,2,2,1,0,
					      0,0,1,0,1,
					      0,2,0,0,1,
					      0,1,2,0,0,
					      2,2,1,0,2,
					      2,1,2,2,0,
					      1,0,1,0,1,
					      0,0,1,0,1,
					      1,0,1,2,2,
					      0,1,0,1,0,
					      0,2,2,1,1,
					      2,1,1,2,0,
					      2,1,0,1,2};
	float filer_w[] = {-1, 0,-1,
					    0, 1, 0,
					    0,-1, 0,
					    0, 1, 0,
					    1, 0, 1,
					    0,-1, 1,
					    1,-1, 1,
					    0, 0, 1,
					    0,-1, 0,
					    1,-1,-1,
					    0,-1, 1,
					    0,-1, 1,
					    1, 1, 0,
					    1, 0, 0,
					    0, 1, 1,
					    1, 0,-1,
					   -1, 1, 1,
					    0,-1, 0};
	convMatrix input(input_data, 2, 5, 5, 3);
	convMatrix filter(filer_w, 3, 3, 3, 2);
	convMatrix target(2, 3, 3, 2);
	input.convolve(filter,2,true, target);
	Matrix flat(2, 3*3*2);
	target.print_data();
	target.flatten(flat);
	for (int i = 0; i < flat.getNumRows(); i++) {
		for (int j = 0; j < flat.getNumCols(); j++) {
			std::cout << flat(i,j) << " ";
		}
		std:: cout << "\n";
	}
}