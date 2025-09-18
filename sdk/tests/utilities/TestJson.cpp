// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "data_types_generated.h"
#include "graph_generated.h"
#include "hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp"
#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Json.hpp>

TEST(TestJson, SerializeGraphAsJson)
{
    std::cout << "---SerializeGraphAsJson---\n";
    auto graphBuilder = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto graphFlatbuffer = graphBuilder.Release();
    auto graph = hipdnn_sdk::data_objects::GetGraph(graphFlatbuffer.data());

    nlohmann::json graphJson = *graph;
    std::string out = graphJson.dump();
    std::cout << out << "\n";

    flatbuffers::FlatBufferBuilder builder;
    auto newGraph = hipdnn_sdk::json::graph(builder, graphJson);
    builder.Finish(newGraph);
    auto finishedGraph = hipdnn_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    std::cout << nlohmann::json(*finishedGraph).dump() << "\n";
}

TEST(TestJson, DataTypeConversion)
{
    {
        nlohmann::json obj = hipdnn_sdk::data_objects::DataType::FLOAT;

        std::cout << obj.dump() << "\n";
    }

    nlohmann::json obj = "float";
    ASSERT_EQ(obj.get<hipdnn_sdk::data_objects::DataType>(),
              hipdnn_sdk::data_objects::DataType::FLOAT);
}
