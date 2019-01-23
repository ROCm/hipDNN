#include "test_activation_common.hpp"

// alpha_beta[4] = {0.f, 0.5f, 1.f, 2.f};

TEST(activation_backward, func_test_activation_RELU) {

  // alpha = 1.f, beta = 0.f;
  activation_params_t test_case(1, 1, 224, 244);
  act_mode = HIPDNN_ACTIVATION_RELU;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_RELU",
                     alpha_beta[2], alpha_beta[0]);
}

TEST(activation_backward, func_test_activation_SIGMOID) {

  // alpha = 1.f, beta = 0.5f;
  activation_params_t test_case(1, 1, 224, 244);
  act_mode = HIPDNN_ACTIVATION_SIGMOID;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_SIGMOID",
                     alpha_beta[2], alpha_beta[1]);
}

TEST(activation_backward, func_test_activation_TANH) {

  activation_params_t test_case(1, 1, 224, 244);
  // alpha = 2.f, beta = 0.f;
  act_mode = HIPDNN_ACTIVATION_TANH;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_TANH",
                     alpha_beta[3], alpha_beta[0]);
}

TEST(activation_backward, func_test_activation_batch32) {

  activation_params_t test_case(32, 1, 224, 244);
  // alpha = 2.f, beta = 1.f;
  act_mode = HIPDNN_ACTIVATION_RELU;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_batch32",
                     alpha_beta[3], alpha_beta[2]);
}

TEST(activation_backward, func_test_activation_batch64) {

  activation_params_t test_case(64, 1, 224, 244);
  // alpha = 0.5f, beta = 0.f;
  act_mode = HIPDNN_ACTIVATION_RELU;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_batch32",
                     alpha_beta[1], alpha_beta[0]);
}

TEST(activation_backward, func_test_activation_batch128) {

  activation_params_t test_case(128, 1, 4, 4);
  // alpha = 2.f, beta = 0.5f;
  act_mode = HIPDNN_ACTIVATION_RELU;

  Test_activation_bwd<float>(test_case, act_mode,
                     "activation_backward:func_test_activation_batch32",
                     alpha_beta[3], alpha_beta[1]);
}