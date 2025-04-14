// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "graph_instance.hpp"

#include <gtest/gtest.h>

TEST(GraphInstanceTest, ConstructorInitializesGraphType)
{
    int            graph_type = 42;
    Graph_instance graph_instance(graph_type);
    EXPECT_EQ(graph_instance.get_graph_type(), graph_type);
}

TEST(GraphInstanceTest, HandlesNegativeGraphType)
{
    int            negative_graph_type = -1;
    Graph_instance graph_instance(negative_graph_type);
    EXPECT_EQ(graph_instance.get_graph_type(), negative_graph_type);
}
