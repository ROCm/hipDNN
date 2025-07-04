// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_manager.hpp"
#include "engines/miopen_engine.hpp"
#include <algorithm>

namespace miopen_legacy_plugin
{

Engine_manager::Engine_manager() {}

void Engine_manager::add_engine(std::unique_ptr<Engine> engine)
{
    _engines.insert(std::move(engine));
}

std::set<int64_t> Engine_manager::get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph)
{
    std::set<int64_t> applicable;
    for(const auto& engine : _engines)
    {
        if(engine->is_applicable(op_graph))
        {
            applicable.insert(engine->id());
        }
    }
    return applicable;
}

}