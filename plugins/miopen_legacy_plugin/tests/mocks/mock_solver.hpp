// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/solvers/solver.hpp"

#include <gmock/gmock.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace miopen_legacy_plugin
{

class Mock_solver : public Solver
{
public:
    MOCK_METHOD(bool,
                is_applicable,
                (const hipdnn_sdk::data_objects::GraphT& op_graph),
                (const, override));
    MOCK_METHOD(size_t, get_workspace_size, (), (const, override));
};

}