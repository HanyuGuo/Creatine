#ifndef MMEMORY_HPP_
#define MMEMORY_HPP_
#include <map>
#include <cuda.h>
#include <string.h>
#include <vector>
#include <assert.h>

#include <helper_cuda.h>

class MemoryManager;
// class CUDADeviceMemoryManager;
// class HostMemoryManager;

/**
 * Base class for managing memory.
 */
class MemoryManager
{
public:
	virtual void InitMem() = 0; // Other classes must overwrite this function for memory alloc
	MemoryManager();
	virtual ~MemoryManager();

};

// /**
//  *
//  */
// class CUDADeviceMemoryManager:public MemoryManager {
// private:
// 	/* data */
//
// public:
// 	CUDADeviceMemoryManager (arguments);
// 	virtual ~CUDADeviceMemoryManager ();
// };

#endif
