// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file plugin_api_enums.h
 * @brief Header file for hipDNN Plugin API Enums
 *
 * This file contains the definitions of the enumerations used in the hipDNN Plugin API.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enumeration for the status of plugin operations.
 *
 * This enumeration defines the different status codes that can be returned by
 * the hipDNN Plugin API functions. The status codes include:
 * - hipdnnPluginStatusSuccess: Indicates success.
 * - hipdnnPluginStatusBadParam: Indicates a bad parameter.
 * - hipdnnPluginInvalidValue: Indicates an invalid value.
 * - hipdnnPluginInternalError: Indicates an internal error.
 */
typedef enum
{
    hipdnnPluginStatusSuccess  = 0,
    hipdnnPluginStatusBadParam = 1,
    hipdnnPluginInvalidValue   = 2,
    hipdnnPluginInternalError  = 3,
} hipdnnPluginStatus_t;

/**
 * @brief Enumeration for the type of plugin.
 *
 * This enumeration defines the different types of plugins that can be created
 * using the hipDNN Plugin API. The types include:
 * - hipdnnPluginTypeUnspecified: Indicates an unspecified plugin type.
 * - hipdnnPluginTypeEngine: Indicates a plugin that has engines.
 */
typedef enum
{
    hipdnnPluginTypeUnspecified = 0,
    hipdnnPluginTypeEngine      = 1,
} hipdnnPluginType_t;

#ifdef __cplusplus
}
#endif
