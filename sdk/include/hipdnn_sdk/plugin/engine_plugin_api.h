// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint> // for uint32_t

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/plugin_api.h>

/**
 * @file engine_plugin_api.h
 * @brief hipDNN Engine Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Engine Plugin API.
 * The API allows users to create and manage custom plugins for hipDNN.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup EnginePluginFunctions Engine Plugin API Functions
 * @brief Functions that each engine plugin must implement.
 * @{
 */

#if 1 // TODO Temporary functions, these are going to be removed soon.
/**
 * @brief Retrieves the number of engines available in the plugin.
 * @param[out] num_engines Pointer to an unsigned integer where the number of engines will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetNumEngines(unsigned* num_engines);

/**
 * @brief Runs the specified engine with the given input.
 * @param[in] engine_index The index of the engine to run.
 * @param[in] input Pointer to the input data to be processed by the engine.
 * @param[out] output Pointer to a buffer where the processed output data will be stored.
 * @param[in] size The size of the input data array.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginRunEngine(unsigned engine_index,
                                                                const uint32_t* input,
                                                                uint32_t* output,
                                                                uint32_t size);
#endif

/**
 * @brief Creates a handle for the engine plugin.
 * @param[out] handle Pointer to a handle that will be created.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle);

/**
 * @brief Destroys the handle for the engine plugin.
 * @param[in] handle The handle to be destroyed.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle);

/**
 * @brief Sets the stream for the engine plugin.
 * @param[out] handle The handle to the engine plugin.
 * @param[in] stream The HIP stream to be used by the plugin.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream);

/**
 * @brief Retrieves the serialized engines from the plugin.
 * @param[in] handle The handle to the engine plugin.
 * @param[in] op_graph Pointer to a structure where the serialized "Graph" from graph.fbs is stored.
 * @param[in] max_engines Limits the maximum number of engines to be returned.
 * @param[in,out] engines Pointer to a structure where the serialized "EngineContainer" from engine.fbs will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 * @note The engines structure is provided by the user, while the function fills in its fields, including allocating
 *       the buffer for the serialized "EngineContainer". After use, this memory must be freed using
 *       hipdnnEnginePluginDestroyEngines().
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngines(hipdnnEnginePluginHandle_t handle,
                                 const hipdnnPluginConstData_t* op_graph,
                                 unsigned max_engines,
                                 hipdnnPluginConstData_t* engines);

/**
 * @brief Destroys the serialized "EngineContainer".
 * @param[in] handle The handle to the engine plugin.
 * @param[in,out] engines Pointer to a structure where the serialized "EngineContainer" from engine.fbs is stored.
 * @return A value of type hipdnnPluginStatus_t.
 * @note The function takes a structure as input, deallocates the buffer, and sets all fields to 0.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngines(
    hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engines);

/**
 * @brief Retrieves the workspace size required for the engine.
 * @param[in] handle The handle to the engine plugin.
 * @param[in] engine_config Pointer to a structure where the serialized "EngineConfig" from engine_config.fbs is stored.
 * @param[in] op_graph Pointer to a structure where the serialized "Graph" from graph.fbs is stored.
 * @param[out] workspace_size Pointer to a size_t variable where the workspace size will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size);

/**
 * @brief Creates a serialized "ExecutionPlan" for the engine config and operation graph.
 * @param[in] handle The handle to the engine plugin.
 * @param[in] engine_config Pointer to a structure where the serialized "EngineConfig" from engine_config.fbs is stored.
 * @param[in] op_graph Pointer to a structure where the serialized "Graph" from graph.fbs is stored.
 * @param[in,out] exec_plan Pointer to a structure where the serialized "ExecutionPlan" from execution_plan.fbs will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 * @note The exec_plan structure is provided by the user, while the function fills in its fields, including allocating
 *       the buffer for the serialized "ExecutionPlan". After use, this memory must be freed using
 *       hipdnnEnginePluginDestroyExecPlan().
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecPlan(hipdnnEnginePluginHandle_t handle,
                                     const hipdnnPluginConstData_t* engine_config,
                                     const hipdnnPluginConstData_t* op_graph,
                                     hipdnnPluginConstData_t* exec_plan);

/**
 * @brief Destroys the serialized "ExecutionPlan".
 * @param[in] handle The handle to the engine plugin.
 * @param[in,out] exec_plan Pointer to a structure where the serialized "ExecutionPlan" from execution_plan.fbs is stored.
 * @return A value of type hipdnnPluginStatus_t.
 * @note The function takes a structure as input, deallocates the buffer, and sets all fields to 0.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecPlan(
    hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* exec_plan);

/**
 * @brief Executes the operation graph using the specified execution plan.
 * @param[in] handle The handle to the engine plugin.
 * @param[in] exec_plan Pointer to a structure where the serialized "ExecutionPlan" from execution_plan.fbs is stored.
 * @param[in] op_graph Pointer to a structure where the serialized "Graph" from graph.fbs is stored.
 * @param[in] device_buffers Pointer to a structure where the serialized "DeviceBuffers" from device_buffers.fbs is stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     const hipdnnPluginConstData_t* exec_plan,
                                     const hipdnnPluginConstData_t* op_graph,
                                     const hipdnnPluginConstData_t* device_buffers);

/** @} */ // End of EnginePluginFunctions group

#ifdef __cplusplus
}
#endif
