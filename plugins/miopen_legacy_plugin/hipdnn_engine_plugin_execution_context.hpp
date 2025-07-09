// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

struct hipdnnEnginePluginExecutionContext
{
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT> engine_config;
};