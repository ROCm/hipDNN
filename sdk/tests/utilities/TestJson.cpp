// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "data_types_generated.h"
#include "graph_generated.h"
#include "hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp"
#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Json.hpp>
#include <spdlog/fmt/bundled/format.h>
#include <type_traits>

using namespace hipdnn_sdk::data_objects;

TEST(TestJson, GraphToJsonAndBack)
{
    auto graphBuilder = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

    nlohmann::json graphJson = *graph;

    flatbuffers::FlatBufferBuilder builder;
    auto newGraphBuilder = hipdnn_sdk::json::toGraph(builder, graphJson);
    builder.Finish(newGraphBuilder);
    auto newGraph = hipdnn_sdk::data_objects::GetGraph(builder.GetBufferPointer());

    EXPECT_EQ(graph->compute_type(), newGraph->compute_type());
    EXPECT_EQ(graph->io_type(), newGraph->io_type());
    EXPECT_EQ(graph->name()->str(), newGraph->name()->str());

    ASSERT_EQ(graph->tensors()->size(), newGraph->tensors()->size());
    auto t1 = graph->tensors()->begin();
    auto t2 = newGraph->tensors()->begin();
    for(; t1 != graph->tensors()->end() && t2 != newGraph->tensors()->end(); t1++, t2++)
    {
        EXPECT_EQ(*t1->UnPack(), *t2->UnPack());
    }

    ASSERT_EQ(graph->nodes()->size(), newGraph->nodes()->size());
    auto n1 = graph->nodes()->begin();
    auto n2 = newGraph->nodes()->begin();
    for(; n1 != graph->nodes()->end() && n2 != newGraph->nodes()->end(); n1++, n2++)
    {
        EXPECT_EQ(*n1->UnPack(), *n2->UnPack());
    }
}

TEST(TestJson, FromVector)
{
    std::vector<int> vec = {0, 1, 2, 3, 4};
    nlohmann::json vecJson = vec;
    EXPECT_EQ(vec.size(), vecJson.size());
    for(size_t i = 0; i < vec.size(); i++)
    {
        ASSERT_EQ(vec[i], vecJson[i].get<int>());
    }
}

template <class T>
void enumTestSuite(T value, const std::string& stringRep, const std::string& context)
{
    auto jsonStringRep = "\"" + stringRep + "\"";
    nlohmann::json jsonValue = value;
    EXPECT_EQ(value, jsonValue.get<T>()) << context;
    EXPECT_EQ(jsonValue.dump(), std::string{jsonStringRep}) << context;
    EXPECT_EQ(nlohmann::json(stringRep).get<T>(), value) << context;
    std::cout << nlohmann::json{stringRep} << "\n";
}

TEST(TestJson, Enum)
{
    using namespace hipdnn_sdk::data_objects;

    enumTestSuite(DataType::FLOAT, "float", "(for hipdnn_sdk::data_objects::DataType)");
    enumTestSuite(NodeAttributes::BatchnormInferenceAttributes,
                  "BatchnormInferenceAttributes",
                  "(for hipdnn_sdk::data_objects::NodeAttributes)");
}
