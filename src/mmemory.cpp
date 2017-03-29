#include <iostream>
#include <assert.h>
#include <cstdlib>

#include "mmemory.hpp"

void * ResourceManager::_init(double * data, long int length, long int width){
    long int length = length*width;
    double id = alloc_id++;
    allocationList[id] = 1;
    std::cout << "id: "<<id<<"\n";
    data = (double *)malloc(sizeof(double)*(length+1));
    data[0] = id;
    return (void *)data;

}


bool ResourceManager::check_mem(double *data) {
   double id = *(data-sizeof(double));
   return allocationList[id]?true:false;
}

ResourceManager::ResourceManager(){
  _init(NULL,0,0);
}

ResourceManager::ResourceManager(double *data, long int length, long int width) {
  _init(data, length,width);
  bool ch = check_mem(data);
  if (ch != 1) {
    std::cout << "Memory already allocated ....exiting...\n";
    exit(1);
  }
}


ResourceManager::~ResourceManager() {

}
