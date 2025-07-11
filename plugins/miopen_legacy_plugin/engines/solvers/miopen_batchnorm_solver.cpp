/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include "miopen_batchnorm_solver.hpp"

namespace miopen_legacy_plugin
{

bool Miopen_batchnorm_solver::is_applicable(const hipdnn_sdk::data_objects::GraphT& op_graph) const
{
    // TODO: Implement logic to check if the solver is applicable to the given op_graph.
    return true;
}

size_t Miopen_batchnorm_solver::get_workspace_size() const
{
    // TODO: Implement logic to determine the required workspace size.
    // Placeholder: return 0.
    return 0;
}

}