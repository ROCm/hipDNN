// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_tensor.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(CpuFpReferenceImplementation, BasicFloatUsage)
{
    Test_tensor input_tensor = Test_tensor::make_test_tensor<float>({1, 3, 224, 224});
    Test_tensor output_tensor = Test_tensor::make_test_tensor<float>({1, 3, 224, 224});
    Test_tensor bias_tensor = Test_tensor::make_test_tensor<float>({1, 3});
    Test_tensor scale_tensor = Test_tensor::make_test_tensor<float>({1, 3});
    Test_tensor mean_tensor = Test_tensor::make_test_tensor<float>({1, 3});
    Test_tensor variance_tensor = Test_tensor::make_test_tensor<float>({1, 3});

    Cpu_fp_reference_implementation<float, float, float> ref_impl;

    ref_impl.execute(input_tensor,
                     scale_tensor,
                     bias_tensor,
                     mean_tensor,
                     variance_tensor,
                     output_tensor,
                     1e-5f);
}
