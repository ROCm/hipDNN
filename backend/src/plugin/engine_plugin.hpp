// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin_core.hpp"
#include <cstdint> // for uint32_t

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin : public Plugin_base
{
public:
    unsigned num_engines() const;
    void run_engine(unsigned engine_index,
                    const uint32_t* input,
                    uint32_t* output,
                    uint32_t size) const;

protected:
    // resolve_symbols must be called before using the plugin. It is used to resolve the symbols in the plugin library.
    bool resolve_symbols() override;

private:
#ifndef NDEBUG
    bool _initialized = false;
#endif

    hipdnnPluginStatus_t (*_func_get_num_engines)(unsigned*);
    hipdnnPluginStatus_t (*_func_run_engine)(unsigned, const uint32_t*, uint32_t*, uint32_t);

    using Plugin_base::Plugin_base;
    friend class Plugin_manager_base<Engine_plugin>;
};

using Engine_plugin_manager = Plugin_manager_base<Engine_plugin>;

} // namespace plugin
} // hipdnn_backend
