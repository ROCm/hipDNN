// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/engine_plugin_resource_manager.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>

struct hipdnnHandle // NOLINT
{
public:
    hipdnnHandle();
    virtual ~hipdnnHandle() = default;
    virtual void set_stream(hipStream_t stream);
    virtual hipStream_t get_stream() const;
    virtual std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager>
        get_plugin_resource_manager() const;

private:
    hipStream_t _stream = nullptr;
    std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager>
        _plugin_resource_manager;
};
