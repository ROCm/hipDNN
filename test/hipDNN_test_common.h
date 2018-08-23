#ifndef HIPDNN_TEST_COMMON_HPP
#define HIPDNN_TEST_COMMON_HPP

#include "hipDNN.h"
#include <cstdlib>
#include <vector>

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb; // mini batches
    int ng; // number of groups
    int ic, ih, iw;  // Input channels, height and width
    int oc, oh, ow;  // Output channels, height and width
    int kh, kw;  // kernel height and width
    int padh, padw; // padding along height and width
    int strh, strw; // stride along height and width
    int dilh, dilw; // dilation along height and width
};





#define checkHIPDNN(expression)                               \
  {                                                          \
    hipdnnStatus_t status = (expression);                     \
    if (status != HIPDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << hipdnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}


#endif

template<typename dataType>
struct Memory {
private:
    std::vector<dataType> hVec;
    dataType* h_data;
    dataType *d_data;
    size_t mem_size =0;
    int num_of_items =0;
public:
    Memory(int numElements) {
        num_of_items = numElements;
        mem_size = sizeof(dataType) * numElements;
        hVec[numElements];
        this->h_data = (dataType*)malloc(mem_size);
        HIP_CALL(hipMalloc(&this->d_data, mem_size));

    }
    dataType* cpu() {
        return this->h_data;
    }
    dataType* gpu() {
        return this->d_data;
    }
    size_t size(){
        return this->mem_size;
    }
    std::vector<dataType> get_vector() {
        return hVec;
    }
    int get_num_elements() {
        return num_of_items;
    }
};


// Note; We are testing only NCHW format for now
struct Desc {
    Desc(int N, int C, int H, int W): N(N), C(C), H(H), W(W) {}
    int N;
    int C;
    int H;
    int W;
};

// Note we are currently only dealing with 2D convolution
Desc calculateConv2DOutputDesc(Desc inputDesc, Desc filterDesc, int pad[2], int stride[2]) {
    assert(inputDesc.C == filterDesc.C);
    int outputHeight = ((inputDesc.H - filterDesc.H + 2 * pad[0]) / 2 * stride[0]) + 1;
    int outputWidth = ((inputDesc.W - filterDesc.W + 2 * pad[1]) / 2 * stride[1]) + 1;
    Desc outputDesc(inputDesc.N, filterDesc.N, outputHeight, outputWidth);
    return outputDesc;
}

template <typename dataType>
Memory<dataType> createMemory(Desc desc) {
    Memory<dataType> m = Memory<dataType>(desc.N * desc.C * desc.H * desc.W);
    return m;
}
