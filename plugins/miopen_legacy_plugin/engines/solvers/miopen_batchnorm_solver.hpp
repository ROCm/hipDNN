// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "solver.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

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
    size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                              const hipdnn_sdk::data_objects::GraphT& graph) const override;

    void execute_graph(const hipdnnEnginePluginHandle& handle,
                       const hipdnn_sdk::data_objects::GraphT& graph,
                       const hipdnnPluginDeviceBuffer_t* device_buffers,
                       uint32_t num_device_buffers,
                       void* workspace = nullptr) const;

private:
    void execute_batchnorm_fwd_inference(
        const hipdnnEnginePluginHandle& handle,
        const hipdnn_sdk::data_objects::GraphT& graph,
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributesT& attributes,
        const hipdnnPluginDeviceBuffer_t* device_buffers,
        uint32_t num_device_buffers) const;
};

}
