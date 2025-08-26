// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/miopen_batchnorm_bwd_plan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

using namespace miopen_legacy_plugin;

TEST(BatchnormBwdParamsTest, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_bwd_graph();
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.get_node(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    Batchnorm_bwd_params params(*attrs, graph.get_tensor_map());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.dy());
    EXPECT_NO_THROW(params.dx());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.dscale());
    EXPECT_NO_THROW(params.dbias());

    // Optional tensors should be present
    auto& mean_opt = params.opt_mean();
    auto& var_opt = params.opt_inv_variance();
    EXPECT_TRUE(mean_opt.has_value());
    EXPECT_TRUE(var_opt.has_value());
    EXPECT_NE(mean_opt.value(), nullptr);
    EXPECT_NE(var_opt.value(), nullptr);
}

TEST(BatchnormBwdParamsTest, HandlesMissingOptionalTensors)
{
    // Create a valid batchnorm graph and remove mean/variance from tensor map
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_bwd_graph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false // Set has_optional_attributes to false
    );
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.get_node(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const auto& tensor_map = graph.get_tensor_map();
    Batchnorm_bwd_params params(*attrs, tensor_map);

    // Optional tensors should not be present
    EXPECT_FALSE(params.opt_mean().has_value());
    EXPECT_FALSE(params.opt_inv_variance().has_value());
}
