// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

#include "engine.hpp"
#include "solvers/solver.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Miopen_engine : public Engine
{
public:
    Miopen_engine(int64_t id);

    int64_t id() const override;

    bool is_applicable(const hipdnnPluginConstData_t* op_graph) const override;
    void get_details(hipdnnPluginConstData_t& details_out) const override;
    size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                              const hipdnnPluginConstData_t* op_graph) const override;

    void add_solver(std::unique_ptr<Solver> solver);

private:
    int64_t _id;
    std::set<std::unique_ptr<Solver>> _solvers;
};

}