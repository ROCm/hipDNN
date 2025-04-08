// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_frontend_hello_world.hpp"
#include <gtest/gtest.h>

#include "graph_generated.h"

using namespace hipdnn::sdk;

TEST(HelloWorldFrontendTest, SayHelloReturnsCorrectMessage)
{
    HelloWorldFrontend helloWorld;
    EXPECT_EQ(helloWorld.sayHello(), "Hello, Frontend");
}

TEST(FlatBuffersFrontendTests, SerializeAndDeserialize)
{
    // Create a FlatBufferBuilder
    flatbuffers::FlatBufferBuilder builder;

    auto graph = CreateGraphDirect(builder, "Graph");
    builder.Finish(graph);

    // Deserialize the FlatBuffer
    auto deserializedGraph = flatbuffers::GetRoot<Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserializedGraph->name()->str(), "Graph");
}