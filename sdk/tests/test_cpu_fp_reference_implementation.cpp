// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(CpuFpReferenceImplementation, BasicFloatUsage)
{
    // std::vector<int64_t> strides = {1, 3, 224, 224}; // always in nchw
    // std::vector<int64_t> dims = {1, 3, 224, 224};

    // auto batchnorm_builder = flatbuffer_test_utils::create_valid_batchnorm_graph(strides, dims);

    // std::map<int64_t, Test_tensor> tensors;

    Cpu_fp_reference_implementation<float, float, float> ref_impl;
}
