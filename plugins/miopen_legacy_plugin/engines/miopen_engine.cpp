// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine.hpp"
#include "solvers/miopen_batchnorm_solver.hpp"

namespace miopen_legacy_plugin
{

Miopen_engine::Miopen_engine(int64_t id, std::set<std::unique_ptr<Solver>>& solvers)
    : _id(id)
    , _solvers(solvers)
{
}

int64_t Miopen_engine::id() const
{
    return _id;
}

bool Miopen_engine::is_applicable(const hipdnnPluginConstData_t* op_graph) const
{

    return true;
}

size_t Miopen_engine::get_workspace_size() const
{
    return 1337;
}

// void Miopen_engine::initialize_solvers()
// {
//     _solvers.insert(std::make_shared<Miopen_batchnorm_solver>());
// }

}