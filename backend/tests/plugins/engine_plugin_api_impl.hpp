// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

void get_all_engine_ids(int64_t* engine_ids, uint32_t max_engines, uint32_t* num_engines);
void get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* op_graph,
                               int64_t* engine_ids,
                               uint32_t max_engines,
                               uint32_t* num_engines);
void check_engine_id_validity(int64_t engine_id); // throw if invalid
void get_engine_details(hipdnnEnginePluginHandle_t handle,
                        int64_t engine_id,
                        const hipdnnPluginConstData_t* op_graph,
                        hipdnnPluginConstData_t* engine_details);
void destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                            hipdnnPluginConstData_t* engine_details);
size_t get_workspace_size(hipdnnEnginePluginHandle_t handle,
                          const hipdnnPluginConstData_t* engine_config,
                          const hipdnnPluginConstData_t* op_graph);
hipdnnEnginePluginExecutionContext_t
    create_execution_context(hipdnnEnginePluginHandle_t handle,
                             const hipdnnPluginConstData_t* engine_config,
                             const hipdnnPluginConstData_t* op_graph);
void destroy_execution_context(hipdnnEnginePluginHandle_t handle,
                               hipdnnEnginePluginExecutionContext_t execution_context);
void execute_op_graph(hipdnnEnginePluginHandle_t handle,
                      hipdnnEnginePluginExecutionContext_t execution_context,
                      void* workspace,
                      const hipdnnPluginDeviceBuffer_t* device_buffers,
                      uint32_t num_device_buffers);
