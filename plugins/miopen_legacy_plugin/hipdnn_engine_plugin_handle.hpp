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
        if(_stream == nullptr)
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         "Stream is not set on the MIOpen handle");
        }
        return _stream;
    }

    std::shared_ptr<miopen_legacy_plugin::Miopen_container> miopen_container;
    miopen_legacy_plugin::Engine_manager& get_engine_manager()
    {
        return miopen_container->get_engine_manager();
    }

private:
    hipStream_t _stream = nullptr;

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
