// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "graph_generated.h"
#include <gtest/gtest.h>

TEST(FlatBuffers, SerializeAndDeserialize)
{
    flatbuffers::FlatBufferBuilder builder;

    auto graph = hipdnn::sdk::CreateGraphDirect(builder, "Graph");
    builder.Finish(graph);

    auto deserialized_graph = flatbuffers::GetRoot<hipdnn::sdk::Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserialized_graph->name()->str(), "Graph");
}
