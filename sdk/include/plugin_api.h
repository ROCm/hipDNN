// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file plugin.h
 * @brief hipDNN Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Plugin API.
 * The API allows users to create and manage custom plugins for hipDNN.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enumeration for the type of plugin.
 *
 * This enumeration defines the different types of plugins that can be created
 * using the hipDNN Plugin API. The types include:
 * - hipdnnPluginTypeEngine: Indicates a plugin that has engines.
 */
typedef enum
{
    hipdnnPluginTypeEngine = 0,
} hipdnnPluginType_t;

// TODO: Add actual plugin API functions and structures here.

#ifdef __cplusplus
}
#endif
