// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "graph_instance.hpp"

Graph_instance::Graph_instance(int graph_type)
    : _graph_type(graph_type)
{
}

int Graph_instance::get_graph_type() const
{
    return _graph_type;
}
