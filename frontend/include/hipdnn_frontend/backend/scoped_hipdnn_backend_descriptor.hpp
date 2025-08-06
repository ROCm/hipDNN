// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

#include <hipdnn_frontend/backend/backend_logging_helpers.hpp>
#include <hipdnn_frontend/backend/backend_wrapper.hpp>
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_frontend
{

class Scoped_hipdnn_backend_descriptor
{
private:
    hipdnnBackendDescriptor_t _descriptor;
    bool _valid;

    // For some reason clang isnt picking up the usage of the two variables correctly.
    // Ive marked them as [[maybe_unused]] to avoid warnings.
    static void log_backend_error([[maybe_unused]] const std::string& error_string,
                                  [[maybe_unused]] const hipdnnStatus_t status)
    {
        std::array<char, 256> backend_err_msg;
        hipdnn_frontend::hipdnn_backend().get_last_error_string(backend_err_msg.data(),
                                                                backend_err_msg.size());
        HIPDNN_LOG_ERROR(
            "{}: {}. Backend error string: {}", error_string, status, backend_err_msg.data());
    }

public:
    Scoped_hipdnn_backend_descriptor()
        : _descriptor(nullptr)
        , _valid(false)
    {
    }

    Scoped_hipdnn_backend_descriptor(hipdnnBackendDescriptor_t descriptor)
        : _descriptor(descriptor)
        , _valid(true)
    {
    }

    explicit Scoped_hipdnn_backend_descriptor(hipdnnBackendDescriptorType_t descriptor_type)
    {
        auto status = hipdnn_backend().backend_create_descriptor(descriptor_type, &_descriptor);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            _descriptor = nullptr;
            log_backend_error("Failed to create backend descriptor", status);
        }
    }

    explicit Scoped_hipdnn_backend_descriptor(const uint8_t* serialized_graph,
                                              size_t graph_byte_size)
    {
        auto status = hipdnn_backend().backend_create_and_deserialize_graph_ext(
            &_descriptor, serialized_graph, graph_byte_size);

        _valid = (status == HIPDNN_STATUS_SUCCESS);

        if(!_valid)
        {
            _descriptor = nullptr;
            log_backend_error("Failed to create and deserialize graph", status);
        }
    }

    ~Scoped_hipdnn_backend_descriptor()
    {
        if(_valid && _descriptor != nullptr)
        {
            auto status = hipdnn_backend().backend_destroy_descriptor(_descriptor);

            if(status != HIPDNN_STATUS_SUCCESS)
            {
                log_backend_error("Failed to destroy backend descriptor", status);
            }
        }
    }

    Scoped_hipdnn_backend_descriptor(const Scoped_hipdnn_backend_descriptor&) = delete;
    Scoped_hipdnn_backend_descriptor& operator=(const Scoped_hipdnn_backend_descriptor&) = delete;

    Scoped_hipdnn_backend_descriptor(Scoped_hipdnn_backend_descriptor&& other) noexcept
        : _descriptor(other._descriptor)
        , _valid(other._valid)
    {
        other._descriptor = nullptr;
        other._valid = false;
    }

    Scoped_hipdnn_backend_descriptor& operator=(Scoped_hipdnn_backend_descriptor&& other) noexcept
    {
        if(this != &other)
        {
            if(_valid && _descriptor != nullptr)
            {
                auto status = hipdnn_backend().backend_destroy_descriptor(_descriptor);
                if(status != HIPDNN_STATUS_SUCCESS)
                {
                    log_backend_error("Failed to destroy backend descriptor during move assignment",
                                      status);
                }
            }
            _descriptor = other._descriptor;
            _valid = other._valid;
            other._descriptor = nullptr;
            other._valid = false;
        }
        return *this;
    }

    bool valid() const
    {
        return _valid;
    }

    const hipdnnBackendDescriptor_t& get() const
    {
        return _descriptor;
    }
};

}
