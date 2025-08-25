// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/engine_plugin_resource_manager.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>

struct hipdnnHandle
{
public:
    hipdnnHandle();
    virtual ~hipdnnHandle() = default;
    virtual void setStream(hipStream_t stream);
    virtual hipStream_t getStream() const;
    virtual std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager>
        getPluginResourceManager() const;

private:
    hipStream_t _stream = nullptr;
    std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager> _pluginResourceManager;
};
