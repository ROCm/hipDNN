#include "test_LRN_common.hpp"

TEST(LRN_fwd, func_check_naive_LRN) {

  Desc inputDesc(1, 3, 6, 6);
  Desc outputDesc(1, 3, 6, 6);

  Test_LRN_fwd<float>(inputDesc, outputDesc, "LRN_fwd:func_check_naive_LRN");
}

TEST(LRN_fwd, func_check_LRN_batch16) {

  Desc inputDesc(16, 3, 60, 60);
  Desc outputDesc(16, 3, 60, 60);

  Test_LRN_fwd<float>(inputDesc, outputDesc, "LRN_fwd:func_check_LRN_batch16");
}

TEST(LRN_fwd, func_check_LRN_batch32) {

  Desc inputDesc(16, 3, 60, 60);
  Desc outputDesc(16, 3, 60, 60);

  Test_LRN_fwd<float>(inputDesc, outputDesc, "LRN_fwd:func_check_LRN_batch32");
}

TEST(LRN_fwd, func_check_LRN_batch64) {

  Desc inputDesc(64, 3, 6, 6);
  Desc outputDesc(64, 3, 6, 6);

  Test_LRN_fwd<float>(inputDesc, outputDesc, "LRN_fwd:func_check_LRN_batch64");
}

TEST(LRN_fwd, func_check_LRN_batch128) {

  Desc inputDesc(128, 3, 6, 6);
  Desc outputDesc(128, 3, 6, 6);

  Test_LRN_fwd<float>(inputDesc, outputDesc, "LRN_fwd:func_check_LRN_batch128");
}