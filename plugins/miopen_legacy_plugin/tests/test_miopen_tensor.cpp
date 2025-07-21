// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_tensor.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

using namespace miopen_legacy_plugin;

TEST(MiopenTensorTest, CanCreateAndDestroyTensor)
{
    // Use a real tensor attributes from a valid batchnorm graph
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the first tensor attributes from the tensor map
    const auto& tensor_map = graph.get_tensor_map();
    ASSERT_FALSE(tensor_map.empty());
    const auto* tensor_attr = tensor_map.begin()->second;
    ASSERT_NE(tensor_attr, nullptr);

    // Construct and destroy Miopen_tensor
    EXPECT_NO_THROW({
        Miopen_tensor tensor(*tensor_attr);
        EXPECT_EQ(tensor.uid(), tensor_attr->uid());
        EXPECT_NE(tensor.tensor_descriptor(), nullptr);
    });
}

TEST(MiopenTensorTest, TensorDescriptorIsValid)
{
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& tensor_map = graph.get_tensor_map();
    ASSERT_FALSE(tensor_map.empty());
    const auto* tensor_attr = tensor_map.begin()->second;
    Miopen_tensor tensor(*tensor_attr);

    // The descriptor should be non-null and can be used in MIOpen API calls
    EXPECT_NE(tensor.tensor_descriptor(), nullptr);
}
