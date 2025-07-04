/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#include "miopen_container.hpp"
#include "engine_manager.hpp"
#include "engines/miopen_engine.hpp"
#include "engines/solvers/miopen_batchnorm_solver.hpp"
#include <hipdnn_sdk/logging/logger.hpp>

namespace miopen_legacy_plugin
{

Miopen_container::Miopen_container()
{
    HIPDNN_LOG_INFO("Creating Miopen_container");
    create_solvers();
    create_engines();

    _engine_manager = std::make_unique<Engine_manager>(_engines);
}

Miopen_container::~Miopen_container()
{
    HIPDNN_LOG_INFO("Destroying Miopen_container");
}

void Miopen_container::create_solvers()
{
    _miopen_solvers.insert(std::make_unique<Miopen_batchnorm_solver>());
}
void Miopen_container::create_engines()
{
    int64_t engine_id = 1;

    _engines.insert(std::make_unique<Miopen_engine>(engine_id++, _miopen_solvers));
}

Engine_manager& Miopen_container::get_engine_manager()
{
    return *_engine_manager;
}

} // namespace miopen_legacy_plugin