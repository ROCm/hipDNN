// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>

#include "EnginePlugin.hpp"
#include "HipdnnException.hpp"
#include "PluginCore.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class EnginePluginManager : public PluginManagerBase<EnginePlugin>
{
public:
    EnginePluginManager()
        : PluginManagerBase<EnginePlugin>(getPluginSearchPaths())
    {
    }

private:
    void validateBeforeAdding(const EnginePlugin& plugin) override
    {
        auto engineIds = plugin.getAllEngineIds();
        for(const auto id : engineIds)
        {
            if(_engineIds.contains(id))
            {
                throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                      "Engine ID " + std::to_string(id)
                                          + " already exists in the list");
            }
        }
    }

    void actionAfterAdding(const EnginePlugin& plugin) override
    {
        auto engineIds = plugin.getAllEngineIds();
        _engineIds.insert(engineIds.begin(), engineIds.end());
    }

    std::set<int64_t> _engineIds;

    // TODO: fix
    static std::set<std::filesystem::path> getPluginSearchPaths()
    {
        const char* envPath = std::getenv("HIPDNN_PLUGIN_DIR");
        if(envPath != nullptr && std::string(envPath).length() > 0)
        {
            return {std::filesystem::path(envPath)};
        }

        return {std::filesystem::path("hipdnn_plugins/engines/")};
    }
};

} // namespace plugin
} // namespace hipdnn_backend
