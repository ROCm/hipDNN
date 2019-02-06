#include "test_convolution_common.hpp"

// alpha_beta[4] = {0.f, 0.5f, 1.f, 2.f};

TEST(convolution_bwd_filter, func_check_backward_conv_filter) {

  Desc inputDesc(1, 3, 30, 30);
  Desc filterDesc(1, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};

  // alpha = 1.f, beta = 0.f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
                    "convolution_bwd_filter:func_check_backward_conv_filter",
                    alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch32) {

  Desc inputDesc(32, 3, 10, 10);
  Desc filterDesc(32, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};

  // alpha = 1.f, beta = 0.5f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
                  "convolution_bwd_filter:func_backward_conv_filter_batch32",
                  alpha_beta[2], alpha_beta[1]);
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch64) {

  Desc inputDesc(64, 3, 10, 10);
  Desc filterDesc(64, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};

  // alpha = 1.f, beta = 0.f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
                  "convolution_bwd_filter:func_backward_conv_filter_batch64",
                  alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch128) {

  Desc inputDesc(128, 3, 30, 30);
  Desc filterDesc(3, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};

  // alpha = 1.f, beta = 0.f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
                 "convolution_bwd_filter:func_backward_conv_filter_batch128",
                 alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_bwd_filter, func_check_rectangular_height_smaller) {

  Desc inputDesc(1, 3, 10, 10);
  Desc filterDesc(1, 3, 5, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};

  // alpha = 1.f, beta = 0.f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
              "convolution_bwd_filter:func_check_rectangular_height_smaller",
              alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_bwd_filter, func_check_rectangular_dims_width_smaller) {

  Desc inputDesc(32, 3, 100, 224);
  Desc filterDesc(21, 3, 50, 60);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {2, 2}; // stride 2
  int dil[2] = {1, 1};

  // alpha = 1.f, beta = 0.f;

  Test_convolution_bwd_filter<float>(inputDesc, filterDesc, pad, stride, dil,
          "convolution_bwd_filter:func_check_rectangular_dims_width_smaller",
          alpha_beta[2], alpha_beta[0]);
}