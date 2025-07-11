// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>

#include "engines/solvers/solver_interface.hpp"

namespace miopen_legacy_plugin
{

class Mock_solver : public Solver_interface
{
public:
    MOCK_METHOD(bool,
                is_applicable,
                (const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));

    MOCK_METHOD(void,
                execute_graph,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnnEnginePluginExecutionContext& execution_context,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers,
                 void* workspace),
                (override));
};

}