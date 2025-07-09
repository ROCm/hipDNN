// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.

#pragma once

#include <string>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

extern const char* const PLUGIN_NAME;
extern const char* const PLUGIN_VERSION;
extern const hipdnnPluginType_t PLUGIN_TYPE;

void set_last_error_string(const std::string& error);
