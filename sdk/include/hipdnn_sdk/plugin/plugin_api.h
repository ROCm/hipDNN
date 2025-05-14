// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint> // for uint32_t

#include <hipdnn_sdk/plugin/plugin_api_enums.h>

#ifdef _WIN32
#define HIPDNN_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define HIPDNN_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#error "Unsupported platform or compiler"
#endif

/**
 * @file plugin_api.h
 * @brief hipDNN Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Plugin API.
 * The API allows users to create and manage custom plugins for hipDNN.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PluginFunctions Plugin API Functions
 * @brief Functions that every plugin must implement.
 * @{
 */

/**
 * @brief Retrieves the name of the plugin.
 * @param[out] name Pointer to a constant character pointer where the plugin name will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name);

/**
 * @brief Retrieves the version of the plugin.
 * @param[out] version Pointer to a constant character pointer where the plugin version will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version);

/**
 * @brief Retrieves the type of the plugin.
 * @param[out] type Pointer to a hipdnnPluginType_t where the plugin type will be stored.
 * @return A value of type hipdnnPluginStatus_t.
 */
HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type);

/** @} */ // End of PluginFunctions group

/**
 * @defgroup EnginePluginFunctions Engine Plugin API Functions
 * @brief Functions that each engine plugin must implement.
 * @{
 */

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

/** @} */ // End of EnginePluginFunctions group

#ifdef __cplusplus
}
#endif
