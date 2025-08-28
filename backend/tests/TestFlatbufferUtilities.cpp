// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include "descriptors/TestMacros.hpp"
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{
namespace testing
{

using namespace hipdnn_sdk::data_objects;

class FlatbufferUtilitiesTest : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder createValidGraph()
    {
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensorAttributes;
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        flatbuffers::FlatBufferBuilder builder;
        auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                       "test",
                                                                       DataType_FLOAT,
                                                                       DataType_HALF,
                                                                       DataType_BFLOAT16,
                                                                       &tensorAttributes,
                                                                       &nodes);
        builder.Finish(graphOffset);
        return builder;
    }

    static void verifyGraph(const hipdnn_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_type, DataType_FLOAT);
        EXPECT_EQ(graph.intermediate_type, DataType_HALF);
        EXPECT_EQ(graph.io_type, DataType_BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 0);
        EXPECT_EQ(graph.nodes.size(), 0);
    }
};

TEST_F(FlatbufferUtilitiesTest, WillCorrectlyUnpackValidGraphBuffer)
{
    auto builder = createValidGraph();

    auto serializedGraph = builder.Release();
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_NO_THROW(flatbuffer_utilities::convertSerializedGraphToGraph(
        serializedGraph.data(), serializedGraph.size(), graph));

    verifyGraph(*graph);
}

TEST_F(FlatbufferUtilitiesTest, WillStillHaveValidGraphAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    {
        auto builder = createValidGraph();

        auto serializedGraph = builder.Release();
        ASSERT_NO_THROW(flatbuffer_utilities::convertSerializedGraphToGraph(
            serializedGraph.data(), serializedGraph.size(), graph));
    }

    verifyGraph(*graph);
}

TEST(FlatbufferInvalidTests, WillNotUnpackNullBuffer)
{
    auto [buffer, size] = std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10));

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

TEST(FlatbufferInvalidTests, WillNotUnpackInvalidBuffer)
{
    auto arr = std::array<uint8_t, 10>{0};
    auto [buffer, size] = std::make_pair(arr.data(), size_t(10));

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

TEST(FlatbufferInvalidTests, WillNotUnpackWrongSizeBuffer)
{
    auto builder = FlatbufferUtilitiesTest::createValidGraph();
    auto serializedGraph = builder.Release();
    auto [buffer, size] = std::make_pair(serializedGraph.data(), serializedGraph.size() - 20);

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_STATUS(
        flatbuffer_utilities::convertSerializedGraphToGraph(buffer, size, graph),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

} // namespace testing
} // namespace hipdnn_backend
