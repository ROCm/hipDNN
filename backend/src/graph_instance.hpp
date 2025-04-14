// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GRAPH_INSTANCE_HPP
#define GRAPH_INSTANCE_HPP

#include <string>

class Graph_instance
{
public:
    explicit Graph_instance(int graph_type);

    int get_graph_type() const;

private:
    int _graph_type;
};

#endif // GRAPH_INSTANCE_HPP