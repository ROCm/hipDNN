// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "solver.hpp"

namespace miopen_legacy_plugin
{

class Miopen_batchnorm_solver : public Solver
{
public:
    Miopen_batchnorm_solver() = default;
    ~Miopen_batchnorm_solver() override = default;

    // Disallow copy and assignment
    Miopen_batchnorm_solver(const Miopen_batchnorm_solver&) = delete;
    Miopen_batchnorm_solver& operator=(const Miopen_batchnorm_solver&) = delete;

    bool is_applicable(const hipdnn_sdk::data_objects::GraphT& op_graph) const override;
    size_t get_workspace_size() const override;
};

}
