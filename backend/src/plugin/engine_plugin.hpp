// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint> // for uint32_t

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin : public Plugin_base
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    Engine_plugin(Shared_library&& lib);

public:
    unsigned num_engines() const;
    void run_engine(unsigned engine_index,
                    const uint32_t* input,
                    uint32_t* output,
                    uint32_t size) const;

private:
    void resolve_symbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    hipdnnPluginStatus_t (*_func_get_num_engines)(unsigned*);
    hipdnnPluginStatus_t (*_func_run_engine)(unsigned, const uint32_t*, uint32_t*, uint32_t);

    friend class Plugin_manager_base<Engine_plugin>;
};

using Engine_plugin_manager = Plugin_manager_base<Engine_plugin>;

} // namespace plugin
} // hipdnn_backend
