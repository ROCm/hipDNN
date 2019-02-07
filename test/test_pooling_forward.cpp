#include "test_pooling_common.hpp"

TEST(pooling_fwd, func_check_zero_padding) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_zero_padding", 1.f, 0.5f);

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_zero_padding_HALF", 1.f, 0.5f);

}

TEST(pooling_fwd, func_check_AVERAGE_COUNT_INCLUDE_PADDING) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {2, 2};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_AVERAGE_COUNT_INCLUDE_PADDING", 2.f, 0.f,
              HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_AVERAGE_COUNT_INCLUDE_PADDING_HALF", 2.f, 0.f,
              HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
}

TEST(pooling_fwd, func_check_batch32) {

  Desc inputDesc(32, 1, 224, 224);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                         "pooling_fwd:func_check_batch32", 2.f, 1.f);
  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
                         "pooling_fwd:func_check_batch32_HALF", 2.f, 1.f);

}

TEST(pooling_fwd, func_check_batch64) {

  Desc inputDesc(64, 1, 224, 224);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch64", 2.f, 0.5f);

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch64_HALF", 2.f, 0.5f);
}

TEST(pooling_fwd, func_check_batch128) {

  Desc inputDesc(128, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch128");

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch128_HALF");
}

TEST(pooling_fwd, func_check_rectangular_dims_small_size) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 1};
  int stride[2] = {1, 1};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_small_size");

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_small_size_HALF");

}

TEST(pooling_fwd, func_check_rectangular_dims_medium_size) {

  Desc inputDesc(32, 1, 64, 64);
  int spatial_ext[2] = {4, 2};
  int stride[2] = {1, 1};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_medium_size");

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_medium_size_HALF");
}
