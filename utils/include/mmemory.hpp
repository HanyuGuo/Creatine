#ifndef MMEMORY_HPP_
#define MMEMORY_HPP_
#include <map>
#include <cstdlib>
#include <cuda.h>
#include <string.h>
#include <vector>
#include <assert.h>

#include <helper_cuda.h>

class ResourceManager {
private:
	void _init(double * data, long int length, long int width);
	double* data;
	bool check_mem(double *data); // given an alloc Id check if it's zero or not.
	static std::atomic<double>alloc_id = 0;
	static std::std::vector<double> allocationList = 0;

public:
	ResourceManager ();
	ResourceManager(double *data, long int length,long int width);
	void get_mem_status();

	virtual ~ResourceManager ();

};
// class HostMemoryManager {
// private:
// 	 size_t _size;
// 	 bool _isMemFull;
// public:
// 	 HostMemoryManager();
// 	 void hostAlloc(const size_t &size); // use cudaHostAlloc for pinned memory
// 	 void hostFree();
// 	 size_t get_size();
// 	 void set_size(size_t);
//
//    virtual ~HostMemoryManager();
// };
//
//
// class CUDAMemManager {
// private:
// 	size_t _size;
// 	bool _isGPUBusy;
//
// public:
//  CUDAMemManager();
// 	size_t get_size();
// 	void set_size(size_t);
// 	void deviceQuery(); // function to query the system about the GPUs.
// 	void cudaMalloc();
// 	virtual ~CUDAMemManager();
// };




#endif
