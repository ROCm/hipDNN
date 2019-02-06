#include "test_pooling_common.hpp"

TEST(pooling_fwd, func_check_zero_padding) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_zero_padding", 1.f, 0.5f);
}

TEST(pooling_fwd, func_check_AVERAGE_COUNT_INCLUDE_PADDING) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {2, 2};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_AVERAGE_COUNT_INCLUDE_PADDING", 2.f, 0.f,
              HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
}

TEST(pooling_fwd, func_check_batch32) {

  Desc inputDesc(32, 1, 224, 224);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                         "pooling_fwd:func_check_batch32", 2.f, 1.f);

}

TEST(pooling_fwd, func_check_batch64) {

  Desc inputDesc(64, 1, 224, 224);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch64", 2.f, 0.5f);
}

TEST(pooling_fwd, func_check_batch128) {

  Desc inputDesc(128, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
                          "pooling_fwd:func_check_batch128");
}

TEST(pooling_fwd, func_check_rectangular_dims_small_size) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 1};
  int stride[2] = {1, 1};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_small_size");
}

TEST(pooling_fwd, func_check_rectangular_dims_medium_size) {

  Desc inputDesc(32, 1, 64, 64);
  int spatial_ext[2] = {4, 2};
  int stride[2] = {1, 1};
  int pad[2] = {0,0};

  Test_pooling_fwd<float>(inputDesc, spatial_ext, stride, pad,
              "pooling_fwd:func_check_rectangular_dims_medium_size");
}

/*
//half2

TEST(pooling_fwd, func_check_half) {

  Desc inputDesc(1, 1, 4, 4);
  dataType = HIPDNN_DATA_HALF;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Test_pooling_fwd<half>(inputDesc, spatial_ext, stride, pad,
         "pooling_fwd:func_check_half", 1.f, 0.f, HIPDNN_POOLING_MAX, dataType);
} */
