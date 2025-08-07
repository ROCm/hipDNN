/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#include <hipdnn_sdk/logging/logger.hpp>

#include "engine_manager.hpp"
#include "engines/miopen_engine.hpp"
#include "engines/plans/miopen_batchnorm_plan_builder.hpp"
#include "miopen_container.hpp"

namespace miopen_legacy_plugin
{

Miopen_container::Miopen_container()
{
    HIPDNN_LOG_INFO("Creating Miopen_container");

    int64_t engine_id = 1;
    auto miopen_engine = std::make_unique<Miopen_engine>(engine_id++);

    auto batchnorm_plan_builder = std::make_unique<Miopen_batchnorm_plan_builder>();
    miopen_engine->add_plan_builder(std::move(batchnorm_plan_builder));

    _engine_manager = std::make_unique<Engine_manager>();
    _engine_manager->add_engine(std::move(miopen_engine));
}

Miopen_container::~Miopen_container()
{
    HIPDNN_LOG_INFO("Destroying Miopen_container");
}

Engine_manager& Miopen_container::get_engine_manager()
{
    return *_engine_manager;
}

} // namespace miopen_legacy_plugin
