#include "test_activation_common.hpp"

hipdnnActivationMode_t act_mode;

// alpha_beta[4] = {0.f, 0.5f, 1.f, 2.f}

TEST(activation_forward, func_test_fwd_activation_RELU) {

  // alpha = 1.f, beta = 0.f;
  act_mode = HIPDNN_ACTIVATION_RELU;
  activation_params_t test_case(1, 1, 224, 224);

  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_test_fwd_activation_RELU",
                  alpha_beta[2], alpha_beta[0]);
}

TEST(activation_forward, func_test_fwd_activation_SIGMOID) {

  // alpha = 2.f, beta = 1.f;
  act_mode = HIPDNN_ACTIVATION_SIGMOID;
  activation_params_t test_case(1, 1, 224, 224);

  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_test_fwd_activation_SIGMOID",
                  alpha_beta[3], alpha_beta[2]);
}

TEST(activation_forward, func_test_fwd_activation_TANH) {

  // alpha = 1.f, beta = 0.5f;
  act_mode = HIPDNN_ACTIVATION_TANH;
  activation_params_t test_case(1, 1, 224, 224);
  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_test_fwd_activation_TANH",
                  alpha_beta[2], alpha_beta[1]);
}

TEST(activation_forward, func_fwd_activation_batch32) {

  // alpha = 2.f, beta = 0.f;
  act_mode = HIPDNN_ACTIVATION_RELU;
  activation_params_t test_case(32, 1, 4, 4);
  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_fwd_activation_batch32",
                  alpha_beta[3], alpha_beta[0]);
}

TEST(activation_forward, func_fwd_activation_batch64) {

  // alpha = 0.5f, beta = 0.f;
  act_mode = HIPDNN_ACTIVATION_RELU;
  activation_params_t test_case(64, 1, 4, 4);
  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_fwd_activation_batch64",
                  alpha_beta[1], alpha_beta[0]);
}

TEST(activation_forward, func_fwd_activation_batch128) {

  // alpha = 1.f, beta = 0.5f;
  act_mode = HIPDNN_ACTIVATION_RELU;
  activation_params_t test_case(128, 1, 32, 32);
  Test_activation_fwd<float>(test_case, act_mode,
                  "activation_forward:func_fwd_activation_batch128",
                  alpha_beta[2], alpha_beta[1]);
}