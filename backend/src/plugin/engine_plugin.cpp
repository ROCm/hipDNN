// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_plugin.hpp"

namespace hipdnn_backend
{
namespace plugin
{
bool Engine_plugin::resolve_symbols()
{
    if(!Plugin::resolve_symbols())
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
    _func_get_num_engines = _lib.get<unsigned()>(func_name_get_num_engines);

    const auto func_name_run_engine = "hipdnnPluginRunEngine";
    if(!_lib.has(func_name_run_engine))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_run_engine << "() not found\n";
        return false;
    }
    _func_run_engine = _lib.get<int(unsigned, int)>(func_name_run_engine);

#ifndef NDEBUG
    _initialized = true;
#endif
    return true;
}

unsigned Engine_plugin::num_engines() const
{
    assert(_initialized);
    return _func_get_num_engines();
}

int Engine_plugin::run_engine(unsigned engine_index, int input) const
{
    assert(_initialized);
    return _func_run_engine(engine_index, input);
}

} // namespace plugin
} // hipdnn_backend
