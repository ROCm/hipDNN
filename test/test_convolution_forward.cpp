#include "test_convolution_common.hpp"

float alpha;  //Remove later, now used in other cpp files.
float beta;
float alpha_beta[4] = {0.f, 0.5f, 1.f, 2.f};

TEST(convolution_fwd, func_check_zero_padding_medium_input_batch32) {

  Desc inputDesc(32, 3, 224, 224);
  Desc filterDesc(21, 3, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
// alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                 "convolution_fwd:func_check_zero_padding_medium_input_batch32",
                 alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_zero_padding_small_input_batch32) {

  Desc inputDesc(32, 12, 224, 224);
  Desc filterDesc(4, 12, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
//  alpha = 1.f, beta = 0.5f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                  "convolution_fwd:func_check_zero_padding_small_input_batch32",
                  alpha_beta[2], alpha_beta[1]);
}

TEST(convolution_fwd, func_check_two_strides_medium_kernelsize) {

  Desc inputDesc(1, 3, 600, 600);
  Desc filterDesc(2, 3, 30, 30);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {2, 2}; // stride 2
  int dil[2] = {1, 1};
  // alpha = 0.5f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                     "convolution_fwd:func_check_two_strides_medium_kernelsize",
                     alpha_beta[1], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_padding_and_strides_small_size) {

  Desc inputDesc(1, 3, 6, 6);
  Desc filterDesc(1, 3, 2, 2);

  int pad[2] = {1, 1}; // padding 1
  int stride[2] = {2, 2}; // stride 2
  int dil[2] = {1, 1};
  // alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                    "convolution_fwd:func_check_padding_and_strides_small_size",
                    alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_full_conv) {

  Desc inputDesc(1, 3, 7, 7);
  Desc filterDesc(3, 3, 7, 7);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
  //  alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
          "convolution_fwd:func_check_full_conv", alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_batch64) {

  Desc inputDesc(64, 3, 7, 7);
  Desc filterDesc(3, 3, 3, 3);

  int pad[2] = {2, 2}; // padding 2
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
  // alpha = 0.f, beta = 0.5f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
            "convolution_fwd:func_check_batch64", alpha_beta[0], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_zero_padding_medium_input_batch128) {

  Desc inputDesc(128, 3, 16, 16);
  Desc filterDesc(21, 3, 4, 4);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
  // alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
            "convolution_fwd:func_check_zero_padding_medium_input_batch128",
            alpha_beta[2], alpha_beta[0]);
}

TEST(convolution_fwd, func_check_dilation2x2_batch8) {

  Desc inputDesc(8, 3, 7, 7);
  Desc filterDesc(3, 3, 3, 3);

  int pad[2] = {2, 2}; // padding 2
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {2, 2};
  // alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                 "convolution_fwd:func_check_dilation2x2_batch8", alpha_beta[2],
                 alpha_beta[0]);
}

TEST(convolution_fwd, func_check_dilation3x3_batch8) {

  Desc inputDesc(8, 3, 7, 7);
  Desc filterDesc(3, 3, 3, 3);

  int pad[2] = {2, 2}; // padding 2
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {3, 3};
  // alpha = 2.f, beta = 0.5f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                 "convolution_fwd:func_check_dilation3x3_batch8", alpha_beta[3],
                 alpha_beta[1]);
}

TEST(convolution_fwd, func_check_filter_rectangular_dims_height_smaller) {

  Desc inputDesc(32, 3, 100, 100);
  Desc filterDesc(21, 3, 2, 4);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {2, 2}; // stride 2
  int dil[2] = {1, 1};
  // alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
            "convolution_fwd:func_check_filter_rectangular_dims", alpha_beta[2],
            alpha_beta[0]);
}

TEST(convolution_fwd, func_check_rectangular_dims_width_smaller_large_size) {

  Desc inputDesc(1, 3, 224, 224);
  Desc filterDesc(1, 3, 64, 32);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {4, 4}; // stride 4
  int dil[2] = {1, 1};
  // alpha = 1.f,  beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                   "convolution_fwd:func_check_rectangular_dims_length_smaller",
                   alpha_beta[2], alpha_beta[0]);
}
TEST(convolution_fwd, func_check_rectangular_dims_small_size) {

  Desc inputDesc(32, 3, 10, 5);
  Desc filterDesc(21, 3, 5, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {2, 2}; // stride 2
  int dil[2] = {1, 1};
  // alpha = 1.f, beta = 0.f;

  Test_convolution_fwd<float>( inputDesc, filterDesc, pad, stride, dil,
                  "convolution_fwd:func_check_rectangular_dims_breadth_smaller",
                  alpha_beta[2], alpha_beta[0]);
}