// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_plugin.hpp"

namespace hipdnn_backend
{
namespace plugin
{

bool Engine_plugin::resolve_symbols()
{
    if(!Plugin_base::resolve_symbols())
    {
        return false;
    }

    if(type() != hipdnnPluginTypeEngine)
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Wrong plugin type\n";
        return false;
    }

    const auto func_name_get_num_engines = "hipdnnPluginGetNumEngines";
    if(!_lib.has(func_name_get_num_engines))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_num_engines << "() not found\n";
        return false;
    }
    _func_get_num_engines = _lib.get<hipdnnPluginStatus_t(unsigned*)>(func_name_get_num_engines);

    const auto func_name_run_engine = "hipdnnPluginRunEngine";
    if(!_lib.has(func_name_run_engine))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_run_engine << "() not found\n";
        return false;
    }
    // TODO Fix formatting: fix ugly formatting
    _func_run_engine
        = _lib.get<hipdnnPluginStatus_t(unsigned, const uint32_t*, uint32_t*, uint32_t)>(
            func_name_run_engine);

#ifndef NDEBUG
    _initialized = true;
#endif
    return true;
}

unsigned Engine_plugin::num_engines() const
{
    assert(_initialized);
    // TODO Fix formatting: indentation between the type and the variable name
    unsigned num_engines;
    auto     status = _func_get_num_engines(&num_engines);
    if(status != hipdnnPluginStatusSuccess)
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to get the number of engines. Status code: "
                                 + std::to_string(status));
    }
    return num_engines;
}

// TODO Fix formatting: indentation between the type and the variable name
void Engine_plugin::run_engine(unsigned        engine_index,
                               const uint32_t* input,
                               uint32_t*       output,
                               uint32_t        size) const
{
    assert(_initialized);
    auto status = _func_run_engine(engine_index, input, output, size);
    if(status != hipdnnPluginStatusSuccess)
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to run the engine. Status code: "
                                 + std::to_string(status));
    }
}

} // namespace plugin
} // hipdnn_backend
