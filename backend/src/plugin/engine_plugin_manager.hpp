// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <set>
#include <string>

#include "engine_plugin.hpp"
#include "hipdnn_exception.hpp"
#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin_manager : public Plugin_manager_base<Engine_plugin>
{
public:
    Engine_plugin_manager()
        : Plugin_manager_base<Engine_plugin>({"hipdnn_plugins/engines/"})
    {
    }

private:
    void validate_before_adding(const Engine_plugin& plugin) override
    {
        auto engine_ids = plugin.get_all_engine_ids();
        for(const auto id : engine_ids)
        {
            if(_engine_ids.contains(id))
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                       "Engine ID " + std::to_string(id)
                                           + " already exists in the list");
            }
        }
    }

    void action_after_adding(const Engine_plugin& plugin) override
    {
        auto engine_ids = plugin.get_all_engine_ids();
        _engine_ids.insert(engine_ids.begin(), engine_ids.end());
    }

    std::set<int64_t> _engine_ids;
};

} // namespace plugin
} // namespace hipdnn_backend
