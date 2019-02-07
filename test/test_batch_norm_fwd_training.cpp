#include "test_batchnorm_common.hpp"

TEST(BNorm_Fwd_train, func_check_spatial_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 1, 1, 1);

  std::string testname[5] = {"BNorm_Fwd_train: func_check_spatial_fwd",
                 "BNorm_Fwd_train: func_check_spatial_fwd_resultRunningMean",
                 "BNorm_Fwd_train: func_check_spatial_fwd_resultRunningVariance",
                 "BNorm_Fwd_train: func_check_spatial_fwd_resultSaveMean",
                 "BNorm_Fwd_train: func_check_spatial_fwd_resultSaveVariance"};

  Test_bnorm_fwd_train<float>(inputDesc, outputDesc, bnScaleBiasMeanVarDesc,
                              HIPDNN_BATCHNORM_SPATIAL, testname);

}

TEST(BNorm_Fwd_train, func_check_per_act_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 1, 4, 4);

  std::string testname[5] = {"BNorm_Fwd_train: func_check_per_act_fwd",
                "BNorm_Fwd_train: func_check_per_act_fwd_resultRunningMean",
                "BNorm_Fwd_train: func_check_per_act_fwd_resultRunningVariance",
                "BNorm_Fwd_train: func_check_per_act_fwd_resultSaveMean",
                "BNorm_Fwd_train: func_check_per_act_fwd_resultSaveVariance"};

  Test_bnorm_fwd_train<float>(inputDesc, outputDesc, bnScaleBiasMeanVarDesc,
                              HIPDNN_BATCHNORM_PER_ACTIVATION, testname);

}

TEST(BNorm_Fwd_train, func_check_spatial_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 3, 1, 1);

  std::string testname[5] = {"BNorm_Fwd_train: func_check_spatial_fwd_channel3",
       "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultRunningMean",
       "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultRunningVariance",
       "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultSaveMean",
       "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultSaveVariance"};

  Test_bnorm_fwd_train<float>(inputDesc, outputDesc, bnScaleBiasMeanVarDesc,
                              HIPDNN_BATCHNORM_SPATIAL, testname);
}

TEST(BNorm_Fwd_train, func_check_per_act_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 3, 4, 4);

  std::string testname[5] = {"BNorm_Fwd_train: func_check_per_act_fwd_channel3",
  "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultRunningMean",
  "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultRunningVariance",
  "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultSaveMean",
  "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultSaveVariance"};

  Test_bnorm_fwd_train<float>(inputDesc, outputDesc, bnScaleBiasMeanVarDesc,
                              HIPDNN_BATCHNORM_PER_ACTIVATION, testname);

}