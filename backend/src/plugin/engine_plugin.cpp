// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cassert>

#include "engine_plugin.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Engine_plugin::Engine_plugin(Shared_library&& lib)
    : Plugin_base(std::move(lib))
{
    resolve_symbols();
}

void Engine_plugin::resolve_symbols()
{
    if(type() != hipdnnPluginTypeEngine)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Wrong plugin type");
    }

    const auto func_name_get_num_engines = "hipdnnPluginGetNumEngines";
    _func_get_num_engines
        = _lib.get_symbol<decltype(_func_get_num_engines)>(func_name_get_num_engines);

    const auto func_name_run_engine = "hipdnnPluginRunEngine";
    _func_run_engine = _lib.get_symbol<decltype(_func_run_engine)>(func_name_run_engine);

#ifndef NDEBUG
    _initialized = true;
#endif
}

unsigned Engine_plugin::num_engines() const
{
    assert(_initialized);
    unsigned num_engines;
    auto status = _func_get_num_engines(&num_engines);
    if(status != hipdnnPluginStatusSuccess)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get the number of engines. Status code: "
                                   + std::to_string(status));
    }
    return num_engines;
}

void Engine_plugin::run_engine(unsigned engine_index,
                               const uint32_t* input,
                               uint32_t* output,
                               uint32_t size) const
{
    assert(_initialized);
    auto status = _func_run_engine(engine_index, input, output, size);
    if(status != hipdnnPluginStatusSuccess)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to run the engine. Status code: " + std::to_string(status));
    }
}

} // namespace plugin
} // hipdnn_backend
