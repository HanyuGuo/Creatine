/**
 * Common functionality to be used by the entire framework.
 * Defines lots of static functions to query device and
 * set modes for evaluation.
 * @hanyug: Need to put some more general functions in the class.
 */
#include <iostream>
#include <vector>
#include <cstdlib>


class Creatine {
private:
  bool _gpu_mode, _cpu_mode; // set flags specifying execution state
  Creatine();


public:
  inline static Cmode get_solver_mode(); // get the solver mode.
  inline static void set_solver_mode(Cmode mode); //set the solver mode.
  inline static int getDevice(const int dev_id = 0); // get device id.
  inline static void setDevice(const int dev_id); // set the device.
  // inspired from caffe "include/common.hpp"
  inline static void CheckDevice(); // check if device is available.
  enum Cmode {CPU, GPU}; // possible execution modes for Creatine.
  virtual ~Creatine ();
};
