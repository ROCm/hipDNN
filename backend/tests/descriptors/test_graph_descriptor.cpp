// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/graph_descriptor.hpp"
#include "flatbuffer_test_utils.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "test_macros.hpp"

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
    auto builder = create_valid_graph();
    auto serialized_graph = builder.Release();

    Graph_descriptor descriptor;
    ASSERT_NO_THROW(descriptor.deserialize_graph(serialized_graph.data(), serialized_graph.size()));
    ASSERT_NO_THROW(descriptor.finalize());
}

TEST_F(Graph_descriptor_test, WillFailToSetInvalidGraph)
{
    Graph_descriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.deserialize_graph(nullptr, 0),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(Graph_descriptor_test, FinalizeFailInvalidGraph)
{
    Graph_descriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Graph_descriptor_test, GetAttributeReturnsNotSupported)
{
    Graph_descriptor descriptor;
    int64_t element_count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, &element_count, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Graph_descriptor_test, SetAttributeReturnsNotSupported)
{
    Graph_descriptor descriptor;
    ASSERT_THROW_HIPDNN_STATUS(
        descriptor.set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_DATA_TYPE, 0, nullptr),
        HIPDNN_STATUS_NOT_SUPPORTED);
}
