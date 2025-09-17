// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <any>
#include <cmath>
#include <gtest/gtest.h>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignature.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistrar.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

static BatchnormSignatureKey makeKeyFloat()
{
    return {.inputDataType = DataType::FLOAT,
            .scaleBiasDataType = DataType::FLOAT,
            .meanVarianceDataType = DataType::FLOAT,
            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};
}

static BatchnormSignatureKey makeKeyHalf()
{
    return {.inputDataType = DataType::HALF,
            .scaleBiasDataType = DataType::HALF,
            .meanVarianceDataType = DataType::HALF,
            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};
}

TEST(TestBatchnormSignatureRegistrar, RegistryContainsAllSignatures)
{
    auto& reg = batchnormRegistry();
    EXPECT_EQ(reg.size(), 2u);
    EXPECT_TRUE(reg.contains(makeKeyFloat()));
    EXPECT_TRUE(reg.contains(makeKeyHalf()));
}
