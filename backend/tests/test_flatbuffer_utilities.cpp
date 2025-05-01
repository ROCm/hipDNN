// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "flatbuffer_utilities.hpp"
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{
namespace testing
{

using namespace hipdnn_sdk::data_objects;

class Flatbuffer_utilities_test : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder create_valid_graph()
    {
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensor_attributes;
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        flatbuffers::FlatBufferBuilder builder;
        auto graph_offset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                        "test",
                                                                        DataType_FLOAT,
                                                                        DataType_HALF,
                                                                        DataType_BFLOAT16,
                                                                        &tensor_attributes,
                                                                        &nodes);
        builder.Finish(graph_offset);
        return builder;
    }

    static void verify_graph(const hipdnn_sdk::data_objects::GraphT& graph)
    {
        EXPECT_EQ(graph.name, "test");
        EXPECT_EQ(graph.compute_type, DataType_FLOAT);
        EXPECT_EQ(graph.intermediate_type, DataType_HALF);
        EXPECT_EQ(graph.io_type, DataType_BFLOAT16);
        EXPECT_EQ(graph.tensors.size(), 0);
        EXPECT_EQ(graph.nodes.size(), 0);
    }
};

TEST_F(Flatbuffer_utilities_test, WillCorrectlyUnpackValidGraphBuffer)
{
    auto builder = create_valid_graph();

    auto serialized_graph = builder.Release();
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    auto status = flatbuffer_utilities::convert_serialized_graph_to_graph(
        serialized_graph.data(), serialized_graph.size(), graph);

    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    verify_graph(*graph);
}

TEST_F(Flatbuffer_utilities_test, WillStillHaveValidGraphAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    {
        auto builder = create_valid_graph();

        auto serialized_graph = builder.Release();
        auto status = flatbuffer_utilities::convert_serialized_graph_to_graph(
            serialized_graph.data(), serialized_graph.size(), graph);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    }

    verify_graph(*graph);
}

class Flatbuffer_invalid_tests
    : public Flatbuffer_utilities_test,
      public ::testing::WithParamInterface<std::pair<const uint8_t*, size_t>>
{
};

TEST_P(Flatbuffer_invalid_tests, WillNotUnpackInvalidBuffer)
{
    auto [buffer, size] = GetParam();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    auto status = flatbuffer_utilities::convert_serialized_graph_to_graph(buffer, size, graph);

    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidBufferTests,
    Flatbuffer_invalid_tests,
    ::testing::Values(std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10)),
                      std::make_pair(std::array<uint8_t, 10>{0}.data(), size_t(10)),
                      []() { //Valid graph but incorrect data size
                          auto builder = Flatbuffer_utilities_test::create_valid_graph();
                          auto serialized_graph = builder.Release();
                          return std::make_pair(serialized_graph.data(),
                                                serialized_graph.size() - 20);
                      }()));

} // namespace testing
} // namespace hipdnn_backend
