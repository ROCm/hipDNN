// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engine_plugin.hpp"
#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin_manager: public Plugin_manager_base<Engine_plugin>
{
protected:
    const std::vector<Engine_plugin>& get_plugins() const;
};

} // namespace plugin
} // hipdnn_backend
