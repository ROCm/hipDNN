#ifndef HIPDNN_TEST_COMMON_HPP
#define HIPDNN_TEST_COMMON_HPP

#include "hipDNN.h"
#include "hip/hip_runtime.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <numeric>
#include "csv/csv_integration.hpp"
#include "timer/timer.hpp"

#define BENCHMARK 1

#if BENCHMARK
  int benchmark_iterations = 100;
#else
  int benchmark_iterations = 1;
#endif

#define checkHIPDNN(expression)                                                \
  {                                                                            \
    hipdnnStatus_t status = (expression);                                      \
    if (status != HIPDNN_STATUS_SUCCESS) {                                     \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << hipdnnGetErrorString(status) << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define HIP_CALL(f)                                                            \
  {                                                                            \
    hipError_t err = (f);                                                      \
    if (err != hipSuccess) {                                                   \
      std::cout << "    Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

#endif

template <typename dataType> struct Memory {
private:
  std::vector<dataType> hVec;
  dataType *h_data = NULL;
  dataType *d_data = NULL;
  size_t mem_size = 0;
  int num_of_items = 0;

public:
  Memory(){}
  Memory(int num_of_items) {
    this->num_of_items = num_of_items;
    mem_size = sizeof(dataType) * this->num_of_items;
    this->hVec.reserve(this->num_of_items);
    this->h_data = (dataType *)malloc(this->mem_size);
    HIP_CALL(hipMalloc((void **)&this->d_data, this->mem_size));
  }
  dataType *cpu() { return this->h_data; }

  dataType *gpu() { return this->d_data; }

  size_t size() { return this->mem_size; }

  std::vector<dataType> get_vector() { return this->hVec; }

  int get_num_elements() { return this->num_of_items; }

  ~Memory() {
    assert(this->h_data);
    assert(this->d_data);
    free(this->h_data);
    HIP_CALL(hipFree(this->d_data));
  }

  void printCPUMemory() {
    for (int i = 0; i < num_of_items; i++)
      std::cout << h_data[i] << std::endl;
  }

  void printGPUMemory() {
    dataType *temp = new dataType[this->num_of_items];
    hipMemcpy(temp, d_data, mem_size, hipMemcpyDeviceToHost);
    for (int i = 0; i < num_of_items; i++) {
      std::cout << temp[i] << std::endl;
    }
    delete[] temp;
  }

  dataType *getDataFromGPU() {
    dataType *temp = new dataType[this->num_of_items];
    hipMemcpy(temp, d_data, num_of_items * sizeof(dataType),
              hipMemcpyDeviceToHost);
    return temp;
  }

  dataType *toGPU() {
    hipMemcpy(this->d_data, this->h_data, this->mem_size,
              hipMemcpyHostToDevice);
  }

  dataType *toCPU() {
    hipMempcpy(this->h_data, this->d_data, this->mem_size,
               hipMemcpyDeviceToHost);
  }
};

// Note; We are testing only NCHW format for now
struct Desc {
  Desc(int N, int C, int H, int W) : N(N), C(C), H(H), W(W) {}
  int N;
  int C;
  int H;
  int W;
};

template <typename dataType> Memory<dataType> createMemory(Desc desc) {
  Memory<dataType> m = Memory<dataType>(desc.N * desc.C * desc.H * desc.W);
  return m;
}

template <typename dataType>
void Equals(Memory<dataType> &A, Memory<dataType> &B) {
  // Memcpy the device results to host buffer
  HIP_CALL(hipMemcpy(B.cpu(), B.gpu(), B.size(), hipMemcpyDeviceToHost));
  assert(A.size() == B.size());
  for (int i = 0; i < B.get_num_elements(); i++) {
    EXPECT_NEAR(A.cpu()[i], B.cpu()[i], 0.001);
  }
}

template <typename dataType> void populateMemoryRandom(Memory<dataType> &mem) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{1, 52};
  std::cout << "Creating vector of Size: " << mem.get_num_elements() << std::endl;
  std::vector<dataType> v(mem.get_num_elements());
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::iota(v.begin(), v.end(), 0);
  std::copy(v.begin(), v.end(), mem.cpu());
  // Copy the stuff to device too
  HIP_CALL(hipMemcpy(mem.gpu(), mem.cpu(), mem.size(), hipMemcpyHostToDevice));
}
