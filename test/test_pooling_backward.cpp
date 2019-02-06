#include "test_pooling_common.hpp"

TEST(pooling_backward, func_check_pooling_stride_2x2) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_bwd<float>(inputDesc, spatial_ext, stride, pad,
                  "pooling_backward:func_check_pooling_stride_2x2", 1.f, 0.5f);
}

TEST(pooling_backward, func_check_pooling_AVERAGE_COUNT_INCLUDE_PADDING) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_bwd<float>(inputDesc, spatial_ext, stride, pad,
            "pooling_backward:func_check_pooling_AVERAGE_COUNT_INCLUDE_PADDING",
            1.f, 0.5f, HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
}

TEST(pooling_backward, func_check_pooling_batch32) {

  Desc inputDesc(32, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_bwd<float>(inputDesc, spatial_ext, stride, pad,
            "pooling_backward:func_check_pooling_batch32", 2.f, 0.f);
}

TEST(pooling_backward, func_check_pooling_batch64) {

  Desc inputDesc(64, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_bwd<float>(inputDesc, spatial_ext, stride, pad,
            "pooling_backward:func_check_pooling_batch64",
            1.5, 0.f);
}

TEST(pooling_backward, func_check_pooling_batch128) {

  Desc inputDesc(128, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Test_pooling_bwd<float>(inputDesc, spatial_ext, stride, pad,
            "pooling_backward:func_check_pooling_batch128");

}