#include "../include/convMatrix.hpp"
#include "../include/matrix.hpp"
#include <iostream>

using namespace std;

int main(void){
	
	float input_data[] = {0,  1,  2,  3};
  	float filer_w[] = {0,   1,   2,   3,   4,   5,   6,   7};
   	// float correct[] = {18}
   	convMatrix input(input_data, 1, 2, 2, 1);
	convMatrix filter(filer_w, 2, 2, 1, 2);
	convMatrix target(1, 2, 2, 2);
	input.convolve(filter,1, true, target);
	// Matrix flat(1, 14*14*5);
	// target.print_data();
	// target.flatten(flat);
	// float sum = 0;
	for (int i = 0; i < target.getDim(0); i++) {
		for (int j = 0; j < target.getDim(1); j++) {
			for (int m = 0; m < target.getDim(2); m++) {
				for (int n = 0; n < target.getDim(3); n++) {
					std::cout <<target(i,j,m,n) << " ";
				}
				std:: cout << "\n";
			}
			std:: cout << "\n";
		}
		std:: cout << "\n";
	}
	// std:: cout << sum << "\n";
}