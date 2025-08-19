// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/engine_plugin.hpp"
#include "plugin/engine_plugin_manager.hpp"
#include "plugin/plugin_core.hpp"

#include <filesystem>
#include <gmock/gmock.h>
#include <set>
#include <vector>

namespace hipdnn_backend
{
namespace plugin
{

class Mock_engine_plugin_manager : public Engine_plugin_manager
{
public:
    MOCK_METHOD(void,
                load_plugins,
                (const std::set<std::filesystem::path>& custom_paths,
                 hipdnnPluginLoadingMode_ext_t mode),
                (override));

    MOCK_METHOD(const std::vector<std::shared_ptr<Engine_plugin>>&,
                get_plugins,
                (),
                (const, override));

    MOCK_METHOD(const std::set<std::filesystem::path>&,
                get_loaded_plugin_files,
                (),
                (const, override));
};

} // namespace plugin
} // namespace hipdnn_backend
