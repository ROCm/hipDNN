// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

TEST(FlatBuffers, SerializeAndDeserialize)
{
    flatbuffers::FlatBufferBuilder builder;

    auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(builder, "Graph");
    builder.Finish(graph);

    auto deserialized_graph
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserialized_graph->name()->str(), "Graph");
}
