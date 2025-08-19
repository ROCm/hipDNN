// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin_core.hpp"
#include "engine_plugin.hpp"

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
};

} // namespace plugin
} // namespace hipdnn_backend
