// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_sdk/plugin/test_utils/test_macros.hpp"
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_utilities.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

namespace hipdnn_plugin
{
namespace testing
{

using namespace hipdnn_sdk::data_objects;

class Plugin_flatbuffer_utilities_test : public ::testing::Test
{
public:
    static flatbuffers::FlatBufferBuilder create_valid_batchnorm_graph()
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

TEST_F(Plugin_flatbuffer_utilities_test, WillCorrectlyUnpackValidGraphBuffer)
{
    auto builder = create_valid_batchnorm_graph();

    auto serialized_graph = builder.Release();
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_NO_THROW(flatbuffer_utilities::convert_serialized_plugin_graph_to_graph(
        serialized_graph.data(), serialized_graph.size(), graph));

    verify_graph(*graph);
}

TEST_F(Plugin_flatbuffer_utilities_test, WillStillHaveValidGraphAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    {
        auto builder = create_valid_batchnorm_graph();

        auto serialized_graph = builder.Release();
        ASSERT_NO_THROW(flatbuffer_utilities::convert_serialized_plugin_graph_to_graph(
            serialized_graph.data(), serialized_graph.size(), graph));
    }

    verify_graph(*graph);
}

TEST_F(Plugin_flatbuffer_utilities_test, WillCorrectlyUnpackEngineDetailsBuffer)
{
    auto builder = flatbuffer_test_utils::create_valid_engine_details(1);
    auto serialized_engine_details = builder.Release();
    hipdnnPluginConstData_t engine_details
        = flatbuffer_test_utils::create_valid_const_data_engine_details(serialized_engine_details);

    std::unique_ptr<hipdnn_sdk::data_objects::EngineDetailsT> unpacked_engine_details;
    flatbuffer_utilities::unpack_serialized_engine_details(
        engine_details.ptr, engine_details.size, unpacked_engine_details);
}

TEST_F(Plugin_flatbuffer_utilities_test, WillStillHaveValidEngineDetailsAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_sdk::data_objects::EngineDetailsT> unpacked_engine_details;
    {
        auto builder = flatbuffer_test_utils::create_valid_engine_details(1);
        auto serialized_engine_details = builder.Release();
        hipdnnPluginConstData_t engine_details
            = flatbuffer_test_utils::create_valid_const_data_engine_details(
                serialized_engine_details);

        flatbuffer_utilities::unpack_serialized_engine_details(
            engine_details.ptr, engine_details.size, unpacked_engine_details);
    }

    ASSERT_NE(unpacked_engine_details, nullptr);
    EXPECT_EQ(unpacked_engine_details->engine_id, 1);
}

TEST_F(Plugin_flatbuffer_utilities_test, WillCorrectlyUnpackEngineConfigBuffer)
{
    auto builder = flatbuffer_test_utils::create_valid_engine_config(42);
    auto serialized_engine_config = builder.Release();
    hipdnnPluginConstData_t engine_config
        = flatbuffer_test_utils::create_valid_const_data_engine_config(serialized_engine_config);

    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT> unpacked_engine_config;
    flatbuffer_utilities::unpack_serialized_engine_config(
        engine_config.ptr, engine_config.size, unpacked_engine_config);
    ASSERT_NE(unpacked_engine_config, nullptr);
    EXPECT_EQ(unpacked_engine_config->engine_id, 42);
}

TEST_F(Plugin_flatbuffer_utilities_test, WillStillHaveValidEngineConfigAfterBuilderDestructs)
{
    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT> unpacked_engine_config;
    {
        auto builder = flatbuffer_test_utils::create_valid_engine_config(42);
        auto serialized_engine_config = builder.Release();
        hipdnnPluginConstData_t engine_config
            = flatbuffer_test_utils::create_valid_const_data_engine_config(
                serialized_engine_config);

        flatbuffer_utilities::unpack_serialized_engine_config(
            engine_config.ptr, engine_config.size, unpacked_engine_config);
    }

    ASSERT_NE(unpacked_engine_config, nullptr);
    EXPECT_EQ(unpacked_engine_config->engine_id, 42);
}

class Flatbuffer_invalid_tests
    : public Plugin_flatbuffer_utilities_test,
      public ::testing::WithParamInterface<std::pair<const uint8_t*, size_t>>
{
};

TEST_P(Flatbuffer_invalid_tests, WillNotUnpackInvalidBuffer)
{
    auto [buffer, size] = GetParam();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    ASSERT_THROW_HIPDNN_PLUGIN_STATUS(
        flatbuffer_utilities::convert_serialized_plugin_graph_to_graph(buffer, size, graph),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    ASSERT_EQ(graph, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidBufferTests,
    Flatbuffer_invalid_tests,
    ::testing::Values(std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10)),
                      std::make_pair(std::array<uint8_t, 10>{0}.data(), size_t(10)),
                      []() { //Valid graph but incorrect data size
                          auto builder
                              = Plugin_flatbuffer_utilities_test::create_valid_batchnorm_graph();
                          auto serialized_graph = builder.Release();
                          return std::make_pair(serialized_graph.data(),
                                                serialized_graph.size() - 20);
                      }()));

TEST_P(Flatbuffer_invalid_tests, WillNotUnpackInvalidEngineDetailsBuffer)
{
    auto [buffer, size] = GetParam();

    std::unique_ptr<hipdnn_sdk::data_objects::EngineDetailsT> engine_details;
    ASSERT_THROW_HIPDNN_PLUGIN_STATUS(
        flatbuffer_utilities::unpack_serialized_engine_details(buffer, size, engine_details),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    ASSERT_EQ(engine_details, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidEngineDetailsBufferTests,
    Flatbuffer_invalid_tests,
    ::testing::Values(std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10)),
                      std::make_pair(std::array<uint8_t, 10>{0}.data(), size_t(10)),
                      []() { // Valid engine_details but incorrect data size
                          auto builder = flatbuffer_test_utils::create_valid_engine_details(1);
                          auto serialized_engine_details = builder.Release();
                          return std::make_pair(serialized_engine_details.data(),
                                                serialized_engine_details.size() > 20
                                                    ? serialized_engine_details.size() - 20
                                                    : 0);
                      }()));

TEST_P(Flatbuffer_invalid_tests, WillNotUnpackInvalidEngineConfigBuffer)
{
    auto [buffer, size] = GetParam();

    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT> engine_config;
    ASSERT_THROW_HIPDNN_PLUGIN_STATUS(
        flatbuffer_utilities::unpack_serialized_engine_config(buffer, size, engine_config),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    ASSERT_EQ(engine_config, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidEngineConfigBufferTests,
    Flatbuffer_invalid_tests,
    ::testing::Values(std::make_pair(static_cast<const uint8_t*>(nullptr), size_t(10)),
                      std::make_pair(std::array<uint8_t, 10>{0}.data(), size_t(10)),
                      []() { // Valid engine_config but incorrect data size
                          auto builder = flatbuffer_test_utils::create_valid_engine_config(1);
                          auto serialized_engine_config = builder.Release();
                          return std::make_pair(serialized_engine_config.data(),
                                                serialized_engine_config.size() > 20
                                                    ? serialized_engine_config.size() - 20
                                                    : 0);
                      }()));

} // namespace testing
} // namespace hipdnn_plugin
