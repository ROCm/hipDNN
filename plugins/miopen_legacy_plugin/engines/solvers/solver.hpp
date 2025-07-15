// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace miopen_legacy_plugin
{

class Solver
{
public:
    virtual ~Solver() = default;

    virtual bool is_applicable(const hipdnn_sdk::data_objects::GraphT& op_graph) const = 0;

    virtual size_t get_workspace_size() const = 0;
};

}