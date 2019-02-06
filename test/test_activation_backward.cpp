#include "test_activation_common.hpp"

TEST(activation_backward, func_test_activation_RELU) {

  activation_params_t test_case(1, 1, 224, 244);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_RELU");
}

TEST(activation_backward, func_test_activation_SIGMOID) {

  activation_params_t test_case(1, 1, 224, 244);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_SIGMOID",
                     1.f, 0.5f, HIPDNN_ACTIVATION_SIGMOID);
}

TEST(activation_backward, func_test_activation_TANH) {

  activation_params_t test_case(1, 1, 224, 244);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_TANH",
                     2.f, 0.f, HIPDNN_ACTIVATION_TANH);
}

TEST(activation_backward, func_test_activation_batch32) {

  activation_params_t test_case(32, 1, 224, 244);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_batch32",
                     2.f, 1.f);
}

TEST(activation_backward, func_test_activation_batch64) {

  activation_params_t test_case(64, 1, 224, 244);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_batch32",
                     0.5f, 0.f);
}

TEST(activation_backward, func_test_activation_batch128) {

  activation_params_t test_case(128, 1, 4, 4);

  Test_activation_bwd<float>(test_case,
                     "activation_backward:func_test_activation_batch32",
                     2.f, 0.5f);
}