// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

#include <hipdnn_frontend/backend/backend_wrapper.hpp>
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_frontend
{

class Hipdnn_backend_descriptor
{
private:
    hipdnnBackendDescriptor_t _descriptor;
    bool _valid;

public:
    Hipdnn_backend_descriptor()
        : _descriptor(nullptr)
        , _valid(false)
    {
    }

    explicit Hipdnn_backend_descriptor(hipdnnBackendDescriptorType_t descriptor_type)
    {
        auto status = hipdnn_backend().backend_create_descriptor(descriptor_type, &_descriptor);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            HIPDNN_LOG_ERROR("Failed to create backend descriptor: {}", status);
        }
    }

    explicit Hipdnn_backend_descriptor(const uint8_t* serialized_graph, size_t graph_byte_size)
    {
        auto status = hipdnn_backend().backend_create_and_deserialize_graph_ext(
            &_descriptor, serialized_graph, graph_byte_size);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            HIPDNN_LOG_ERROR("Failed to create and deserialize graph: {}", status);
        }
    }

    ~Hipdnn_backend_descriptor()
    {
        if(_valid && _descriptor != nullptr)
        {
            auto status = hipdnn_backend().backend_destroy_descriptor(_descriptor);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                HIPDNN_LOG_ERROR("Failed to destroy backend descriptor: {}", status);
            }
        }
    }

    bool valid() const
    {
        return _valid;
    }

    hipdnnBackendDescriptor_t get() const
    {
        return _descriptor;
    }
};

}