// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/engine_plugin_resource_manager.hpp"
#include <gmock/gmock.h>

namespace hipdnn_backend
{
namespace plugin
{

class Mock_engine_plugin_resource_manager : public Engine_plugin_resource_manager
{
public:
    MOCK_METHOD(void, set_stream, (hipStream_t stream), (const, override));
    MOCK_METHOD(void,
                execute_op_graph,
                (hipdnnBackendDescriptor_t execution_plan, hipdnnBackendDescriptor_t variant_pack),
                (const, override));
    MOCK_METHOD(std::vector<int64_t>,
                get_applicable_engine_ids,
                (const hipdnn_backend::Graph_descriptor* graph_desc),
                (const, override));
    MOCK_METHOD(void,
                get_engine_details,
                (int64_t engine_id,
                 const hipdnn_backend::Graph_descriptor* graph_desc,
                 hipdnnPluginConstData_t* engine_details),
                (const, override));
    MOCK_METHOD(void,
                destroy_engine_details,
                (int64_t engine_id, hipdnnPluginConstData_t* engine_details),
                (const, override));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (int64_t engine_id,
                 const hipdnnPluginConstData_t* engine_config,
                 const hipdnn_backend::Graph_descriptor* graph_desc),
                (const, override));
    MOCK_METHOD(hipdnnEnginePluginExecutionContext_t,
                create_execution_context,
                (int64_t engine_id,
                 const hipdnnPluginConstData_t* engine_config,
                 const hipdnn_backend::Graph_descriptor* graph_desc),
                (const, override));
    MOCK_METHOD(void,
                destroy_execution_context,
                (int64_t engine_id, hipdnnEnginePluginExecutionContext_t execution_context),
                (const, override));
};

}
}
