#include "../include/gpmatrix.cuh"
#include <iostream>
#include <vector>


int main(int argc, char  *argv[])
{  

	 float *data1 = new double[9];
	 float *data2 = new double[9];
     //double *dev_data1, *dev_data2;
     int num_cols = 3;
     int num_rows = 3;

	 for(int i=0;i<9;++i) {
	 	data1[i] = i;
	 }
	 for(int j=0;j<9;++j) {
	 	data2[j] = 9+j;
	 }
     GpMatrix gp1(data1, 3,3,true);
     GpMatrix gp2(data2, 3,3,true);

     std::cout << "gp1 rows: "<<gp1.getnumRows() << "gp2 rows: "<<gp2.getnumRows()<<"\n";
     std::cout << "gp1 cols: "<<gp2.getnumCols() << "gp2 cols: "<<gp2.getnumCols()<<"\n";
     std::cout<<"gp1 leadingDim: "<<gp1.getLeadingDim()<<"\n";
     std::cout<<"gp2 leadingDim:"<<gp2.getLeadingDim()<<"\n";

    // gp1.add(gp2,1); // simple matrix addition.
	return 0;
}