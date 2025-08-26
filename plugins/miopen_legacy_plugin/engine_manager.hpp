// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_config_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "engines/engine_interface.hpp"

namespace miopen_legacy_plugin
{

class EngineManager
{
public:
    EngineManager() = default;
    ~EngineManager() = default;

    //disallow copy and assignment
    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;

    void addEngine(std::unique_ptr<EngineInterface> engine);

    std::vector<int64_t> getApplicableEngineIds(const hipdnn_plugin::IGraph& opGraph);

    void getEngineDetails(HipdnnEnginePluginHandle& handle,
                          const hipdnn_plugin::IGraph& opGraph,
                          int64_t engineId,
                          hipdnnPluginConstData_t& engineDetailsOut);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            int64_t engineId,
                            const hipdnn_plugin::IGraph& opGraph) const;

    void initializeExecutionContext(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin::IGraph& opGraph,
                                    const hipdnn_plugin::IEngineConfig& engineConfig,
                                    HipdnnEnginePluginExecutionContext& executionContext) const;

private:
    EngineInterface& getEngine(int64_t engineId) const;

    std::unordered_map<int64_t, std::unique_ptr<EngineInterface>> _engines;
};

}
