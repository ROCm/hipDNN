// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/graph_descriptor.hpp"
#include "flatbuffer_test_utils.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

using namespace hipdnn_backend;

class Graph_descriptor_test : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder create_valid_graph()
    {
        return flatbuffer_test_utils::create_valid_graph();
    }

    static void verify_graph(const hipdnn_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
        EXPECT_EQ(graph.intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
        EXPECT_EQ(graph.io_type, hipdnn_sdk::data_objects::DataType_BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 0);
        EXPECT_EQ(graph.nodes.size(), 0);
    }
};

TEST_F(Graph_descriptor_test, WillCorrectlySetGraph)
{
    auto builder          = create_valid_graph();
    auto serialized_graph = builder.Release();

    Graph_descriptor descriptor;
    auto status = descriptor.deserialize_graph(serialized_graph.data(), serialized_graph.size());

    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(Graph_descriptor_test, WillFailToSetInvalidGraph)
{
    Graph_descriptor descriptor;
    auto             status = descriptor.deserialize_graph(nullptr, 0);

    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(Graph_descriptor_test, ExecuteReturnsNotSupported)
{
    Graph_descriptor descriptor;
    auto             status = descriptor.execute(nullptr, nullptr);

    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Graph_descriptor_test, FinalizeReturnsNotSupported)
{
    Graph_descriptor descriptor;
    auto             status = descriptor.finalize();

    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Graph_descriptor_test, GetAttributeReturnsNotSupported)
{
    Graph_descriptor descriptor;
    int64_t          element_count = 0;
    auto             status        = descriptor.get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, &element_count, nullptr);

    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Graph_descriptor_test, SetAttributeReturnsNotSupported)
{
    Graph_descriptor descriptor;
    auto             status
        = descriptor.set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, nullptr);

    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}
