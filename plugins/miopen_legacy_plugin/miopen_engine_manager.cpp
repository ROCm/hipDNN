// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine_manager.hpp"
#include "engines/miopen_fake_engine.hpp"
#include <algorithm>

void Miopen_engine_manager::initialize_engines()
{
    //todo, determine if we want a better way to assign engine ids.  eventually we will probably have
    // the following engines:
    // 1. Miopen_fake_engine (placeholder till we have a real engine)
    // 2. Miopen_conv_engine
    // 3. Miopen_batchnorm_engine
    // 4. Miopen_batchnorm_fusion_engine
    // 5. Miopen_convolution_fusion_engine
    // 6. etc...
    int64_t engine_id = 1;

    _engines.insert(std::make_shared<Miopen_fake_engine>(engine_id++));
}

std::set<int>
    Miopen_engine_manager::get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph)
{
    std::set<int> applicable;
    for(const auto& engine : _engines)
    {
        if(engine->is_applicable(op_graph))
        {
            applicable.insert(engine->id());
        }
    }
    return applicable;
}