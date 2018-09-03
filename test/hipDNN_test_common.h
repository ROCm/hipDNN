#ifndef HIPDNN_TEST_COMMON_HPP
#define HIPDNN_TEST_COMMON_HPP

#include "hipDNN.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#define checkHIPDNN(expression)                                                \
    {                                                                          \
        hipdnnStatus_t status = (expression);                                  \
        if (status != HIPDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "Error on line " << __LINE__ << ": "                  \
                      << hipdnnGetErrorString(status) << std::endl;            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

#define HIP_CALL(f)                                                            \
    {                                                                          \
        hipError_t err = (f);                                                  \
        if (err != hipSuccess) {                                               \
            std::cout << "    Error occurred: " << err << std::endl;           \
            std::exit(1);                                                      \
        }                                                                      \
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
    Memory(int numElements) {
        num_of_items = numElements;
        mem_size = sizeof(dataType) * numElements;
        hVec.reserve(num_of_items);
        this->h_data = (dataType *)malloc(mem_size);
        HIP_CALL(hipMalloc(&this->d_data, mem_size));
    }
    dataType *cpu() { return this->h_data; }
    dataType *gpu() { return this->d_data; }
    size_t size() { return this->mem_size; }
    std::vector<dataType> get_vector() { return hVec; }
    int get_num_elements() { return num_of_items; }
    ~Memory() {
        assert(h_data);
        assert(d_data);
        free(h_data);
        HIP_CALL(hipFree(d_data));
    }
    void printCPUMemory() {
        for (int i = 0; i < num_of_items; i++)
            std::cout << h_data[i] << std::endl;
    }
    void printGPUMemory() {
        dataType *temp = new dataType[num_of_items];
        hipMemcpy(temp, d_data, mem_size, hipMemcpyDeviceToHost);
        for (int i = 0; i < num_of_items; i++) {
            std::cout << temp[i] << std::endl;
        }
        delete[] temp;
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
    printf("Creating vector of Size %d\n", mem.get_num_elements());
    std::vector<dataType> v(mem.get_num_elements());
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(v.begin(), v.end(), gen);
    std::copy(v.begin(), v.end(), mem.cpu());

    // Copy the stuff to device too
    HIP_CALL(
        hipMemcpy(mem.gpu(), mem.cpu(), mem.size(), hipMemcpyHostToDevice));
}
