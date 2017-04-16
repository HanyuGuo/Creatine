#include "../include/utils.hpp"

using namespace std;

int main() {
	char path[] = "../data/layer_1_weight.txt";
	double * weight = new double[2*20];
	load(path, weight);
	for (int i = 0; i < 40; i++)
		cout <<weight[i] << endl;
	delete [] weight;
	return 0;
}