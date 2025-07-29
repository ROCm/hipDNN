// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <miopen/miopen.h>
#include <unordered_map>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

#include "miopen_container.hpp"
#include "miopen_utils.hpp"

// NOLINTBEGIN
struct hipdnnEnginePluginHandle
{
public:
    virtual ~hipdnnEnginePluginHandle() = default;

    miopenHandle_t miopen_handle = nullptr;

    void set_stream(hipStream_t stream)
    {
        THROW_ON_MIOPEN_FAILURE(miopenSetStream(miopen_handle, stream));
        _stream = stream;
    }

    hipStream_t get_stream() const
    {
        return _stream;
    }

    std::shared_ptr<miopen_legacy_plugin::Miopen_container> miopen_container;
    miopen_legacy_plugin::Engine_manager& get_engine_manager()
    {
        return miopen_container->get_engine_manager();
    }

    void store_engine_details_detached_buffer(const void* ptr,
                                              std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_LOG_INFO("Storing detached buffer at address: {:p}", ptr);
        _engine_details_buffers[ptr] = std::move(buffer);
    }

    void remove_engine_details_detached_buffer(const void* ptr)
    {
        HIPDNN_LOG_INFO("Removing detached buffer at address: {:p}", ptr);

        if(_engine_details_buffers.contains(ptr))
        {
            _engine_details_buffers.erase(ptr);
        }
        else
        {
            HIPDNN_LOG_WARN("No detached buffer found at address: {:p}. Could not remove engine "
                            "details. Ensure you "
                            "are using the same hipdnn handle you used for engine details creation",
                            ptr);
        }
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engine_details_buffers;
};

// NOLINTEND
