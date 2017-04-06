#include <iostream>
#include <vector>
#include "../include/gpmatrix.cuh"
#include "../include/gpmatrix_kernels.cuh"


int main(int argc, char  *argv[])
{  

	 double *data1 = new double[9];
	 double *data2 = new double[9];
     double *dev_data1;
     double *dev_data1, *dev_data2;
     int num_cols = 3;
     int num_rows = 3;

	 for(int i=0;i<9;++i) {
	 	data1[i] = i;
	 }
	 for(int j=0;j<9;++j) {
	 	data2[j] = 9+j;
	 }
     GpMatrix gp1 = new GpMatrix(data1, 3,3,true);
     GpMatrix gp2 = new GpMatrix(data2, 3,3,true);

     std::cout << "gp1 rows: "<<gp1.getNumRows() << "gp2 rows: "<<gp2.getNumRows()<<"\n";
     std::cout << "gp1 cols: "<<gp2.getNumCols() << "gp2 cols: "<<gp2.getNumCols()<<"\n";
     std::cout<<"gp1 leadingDim: "<<gp1.getLeadingDim()<<"\n";
     std::cout<<"gp2 leadingDim:"<<gp2.getLeadingDim()<<"\n";

    // gp1.add(gp2,1); // simple matrix addition.
	return 0;
}