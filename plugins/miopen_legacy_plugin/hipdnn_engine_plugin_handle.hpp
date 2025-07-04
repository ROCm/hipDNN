// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <miopen/miopen.h>

#include "miopen_container.hpp"

struct hipdnnEnginePluginHandle
{
public:
    virtual ~hipdnnEnginePluginHandle() = default;

    miopenHandle_t miopen_handle = nullptr;
    hipStream_t stream = nullptr;

    std::shared_ptr<miopen_legacy_plugin::Miopen_container> miopen_container;
    miopen_legacy_plugin::Engine_manager& get_engine_manager()
    {
        return miopen_container->get_engine_manager();
    }
};
