#include "test_batchnorm_common.hpp"

TEST(BNorm_Backward, func_check_spatial_no_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  std::string testname[3] = {"BNorm_Backward: func_check_spatial_no_grad_bwd_dx",
              "BNorm_Backward: func_check_spatial_no_grad_bwd_resultBnScaleDiff",
              "BNorm_Backward: func_check_spatial_no_grad_bwd_resultBnBiasDiff"};

  Test_bnorm_bwd<float>(inputDesc, outputDesc, HIPDNN_BATCHNORM_SPATIAL, 0, testname);
}

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_no_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  std::string testname[3] = {"BNorm_Backward: per_act_mode_no_grad_bwd_dx",
                   "BNorm_Backward: per_act_mode_no_grad_bwd_resultBnScaleDiff",
                   "BNorm_Backward: per_act_mode_no_grad_bwd_resultBnBiasDiff"};

  Test_bnorm_bwd<float>(inputDesc, outputDesc, HIPDNN_BATCHNORM_PER_ACTIVATION, 0, testname);

  }

TEST(BNorm_Backward, func_check_spatial_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  std::string testname[3] = {"BNorm_Backward: func_check_spatial_grad_bwd_dx",
                "BNorm_Backward: func_check_spatial_grad_bwd_resultBnScaleDiff",
                "BNorm_Backward: func_check_spatial_grad_bwd_resultBnBiasDiff"};

  Test_bnorm_bwd<float>(inputDesc, outputDesc, HIPDNN_BATCHNORM_SPATIAL, 1, testname);

  }

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  std::string testname[3] = {"BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_dx",
              "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_resultBnScaleDiff",
              "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_resultBnBiasDiff"};

  Test_bnorm_bwd<float>(inputDesc, outputDesc, HIPDNN_BATCHNORM_PER_ACTIVATION, 1, testname);
}
