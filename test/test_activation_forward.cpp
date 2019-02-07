#include "test_activation_common.hpp"

TEST(activation_forward, func_test_fwd_activation_RELU) {

  activation_params_t test_case(1, 1, 224, 224);

  Test_activation_fwd<float>(test_case,
                            "activation_forward:func_test_fwd_activation_RELU");

  Test_activation_fwd<half>(test_case,
                            "activation_forward:func_test_fwd_activation_RELU_HALF");
}

TEST(activation_forward, func_test_fwd_activation_SIGMOID) {

  activation_params_t test_case(1, 1, 224, 224);

  Test_activation_fwd<float>(test_case,
                  "activation_forward:func_test_fwd_activation_SIGMOID", 2.f,
                  1.f, HIPDNN_ACTIVATION_SIGMOID);

  Test_activation_fwd<half>(test_case,
                  "activation_forward:func_test_fwd_activation_SIGMOID_HALF", 2.f,
                  1.f, HIPDNN_ACTIVATION_SIGMOID);
}

TEST(activation_forward, func_test_fwd_activation_TANH) {

  activation_params_t test_case(1, 1, 224, 224);

  Test_activation_fwd<float>(test_case,
                  "activation_forward:func_test_fwd_activation_TANH", 1.f, 0.f,
                  HIPDNN_ACTIVATION_TANH);

  Test_activation_fwd<half>(test_case,
                  "activation_forward:func_test_fwd_activation_TANH_HALF", 1.f, 0.f,
                  HIPDNN_ACTIVATION_TANH);
}

TEST(activation_forward, func_fwd_activation_batch32) {

  activation_params_t test_case(32, 1, 4, 4);

  Test_activation_fwd<float>(test_case,
                  "activation_forward:func_fwd_activation_batch32", 2.f, 0.f);

  Test_activation_fwd<half>(test_case,
                  "activation_forward:func_fwd_activation_batch32_HALF", 2.f, 0.f);
}

TEST(activation_forward, func_fwd_activation_batch64) {

  activation_params_t test_case(64, 1, 4, 4);

  Test_activation_fwd<float>(test_case,
                  "activation_forward:func_fwd_activation_batch64", 0.5f, 0.f);

  Test_activation_fwd<half>(test_case,
                  "activation_forward:func_fwd_activation_batch64_HALF", 0.5f, 0.f);
}

TEST(activation_forward, func_fwd_activation_batch128) {

  activation_params_t test_case(128, 1, 32, 32);

  Test_activation_fwd<float>(test_case,
                  "activation_forward:func_fwd_activation_batch128", 1.f, 0.5f);

  Test_activation_fwd<half>(test_case,
                  "activation_forward:func_fwd_activation_batch128_HALF", 1.f, 0.5f);
}