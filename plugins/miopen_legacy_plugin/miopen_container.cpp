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

MiopenContainer::MiopenContainer()
{
    HIPDNN_LOG_INFO("Creating MiopenContainer");

    int64_t engine_id = 1;
    auto miopen_engine = std::make_unique<Miopen_engine>(engine_id++);

    auto batchnorm_plan_builder = std::make_unique<Miopen_batchnorm_plan_builder>();
    miopen_engine->add_plan_builder(std::move(batchnorm_plan_builder));

    _engineManager = std::make_unique<EngineManager>();
    _engineManager->addEngine(std::move(miopen_engine));
}

MiopenContainer::~MiopenContainer()
{
    HIPDNN_LOG_INFO("Destroying MiopenContainer");
}

EngineManager& MiopenContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace miopen_legacy_plugin
