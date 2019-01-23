#include "test_convolution_common.hpp"

TEST(convolution_bwd_data, func_check_backward_conv_data) {

  Desc inputDesc(1, 3, 30, 30);
  Desc filterDesc(3, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 1
  int dil[2] = {1,1};

  Test_convolution_bwd_data<float>(inputDesc, filterDesc, pad, stride, dil,
                          "convolution_bwd_data:func_check_backward_conv_data");
}

TEST(convolution_bwd_data, func_bwd_conv_batch8) {

  Desc inputDesc(8, 3, 3, 3);
  Desc filterDesc(8, 3, 2, 2);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};

  Test_convolution_bwd_data<float>(inputDesc, filterDesc, pad, stride, dil,
                          "convolution_bwd_data:func_bwd_conv_batch8");
}

TEST(convolution_bwd_data, func_bwd_conv_batch16) {

  Desc inputDesc(16, 3, 3, 3);
  Desc filterDesc(16, 3, 2, 2);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};

  Test_convolution_bwd_data<float>(inputDesc, filterDesc, pad, stride, dil,
                          "convolution_bwd_data:func_bwd_conv_batch16");
}

TEST(convolution_bwd_data, func_bwd_conv_batch64_pad) {

  Desc inputDesc(64, 3, 3, 3);
  Desc filterDesc(64, 3, 2, 2);

  int pad[2] = {0, 0};
  int stride[2] = {1, 1};
  int dil[2] = {1,1};

  Test_convolution_bwd_data<float>(inputDesc, filterDesc, pad, stride, dil,
                          "convolution_bwd_data:func_bwd_conv_batch64_pad");
}

TEST(convolution_bwd_data, func_bwd_conv_batch128_pad1_stride3) {

  Desc inputDesc(128, 3, 3, 3);
  Desc filterDesc(128, 3, 2, 2);

  int pad[2] = {0, 0};
  int stride[2] = {1, 1};
  int dil[2] = {1,1};

  Test_convolution_bwd_data<float>(inputDesc, filterDesc, pad, stride, dil,
                    "convolution_bwd_data:func_bwd_conv_batch128_pad1_stride3");
}