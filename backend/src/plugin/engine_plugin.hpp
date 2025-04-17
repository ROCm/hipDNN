// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{
// Class for engine plugins
class Engine_plugin : public Plugin
{
public:
    unsigned num_engines() const; // Get the number of engines
    int      run_engine(unsigned engine_index, int input) const; // Run the engine

protected:
    // resolve_symbols must be called before using the plugin. It is used to resolve the symbols in the plugin library.
    bool resolve_symbols() override;

private:
#ifndef NDEBUG
    bool _initialized = false; // Flag to check if the plugin is initialized
#endif

    std::function<unsigned()> _func_get_num_engines; // Function to get the number of engines
    std::function<int(unsigned, int)> _func_run_engine; // Function to run the engine

    using Plugin::Plugin;
    friend class Plugin_manager<Engine_plugin>;
};

// Plugin manager for all engine plugins
using Engine_plugin_manager = Plugin_manager<Engine_plugin>;

} // namespace plugin
} // hipdnn_backend
