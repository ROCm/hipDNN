// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(BatchnormFwdInferenceParamsTest, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = flatbuffer_test_utils::createValidBatchnormGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    BatchnormFwdInferenceParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());

    // Optional tensors should be present
    auto& meanOpt = params.estMean();
    auto& varOpt = params.estVariance();
    EXPECT_TRUE(meanOpt.has_value());
    EXPECT_TRUE(varOpt.has_value());
}

TEST(BatchnormFwdInferenceParamsTest, HandlesMissingOptionalTensors)
{
    // Create a valid batchnorm graph and remove mean/variance from tensor map
    auto builder = flatbuffer_test_utils::createValidBatchnormGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false // Set has_optional_attributes to false
    );
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    const auto& tensorMap = graph.getTensorMap();
    BatchnormFwdInferenceParams params(*attrs, tensorMap);

    // Optional tensors should not be present
    EXPECT_FALSE(params.estMean().has_value());
    EXPECT_FALSE(params.estVariance().has_value());
}
