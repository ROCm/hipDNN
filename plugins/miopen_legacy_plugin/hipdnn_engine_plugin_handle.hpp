// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <miopen/miopen.h>
#include <unordered_map>

#include <hipdnn_sdk/logging/logger.hpp>

#include "miopen_container.hpp"

// NOLINTBEGIN
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

    void store_detached_buffer(const void* ptr, std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_LOG_INFO("Storing detached buffer at address: {:p}", ptr);
        _engine_details_buffers[ptr] = std::move(buffer);
    }

    void remove_detached_buffer(const void* ptr)
    {
        HIPDNN_LOG_INFO("Removing detached buffer at address: {:p}", ptr);

        if(_engine_details_buffers.contains(ptr))
        {
            _engine_details_buffers.erase(ptr);
        }
    }

private:
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engine_details_buffers;
};

// NOLINTEND
