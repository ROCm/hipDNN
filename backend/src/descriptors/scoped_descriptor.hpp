// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_backend.h"

namespace hipdnn_backend
{
struct Scoped_descriptor
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    Scoped_descriptor() = default;
    explicit Scoped_descriptor(hipdnnBackendDescriptor_t desc)
        : descriptor(desc)
    {
    }
    ~Scoped_descriptor()
    {
        delete descriptor;
    }
    Scoped_descriptor(const Scoped_descriptor&) = delete;
    Scoped_descriptor& operator=(const Scoped_descriptor&) = delete;
    Scoped_descriptor(Scoped_descriptor&& other) noexcept
    {
        descriptor = other.descriptor;
        other.descriptor = nullptr;
    }
    Scoped_descriptor& operator=(Scoped_descriptor&& other) noexcept
    {
        if(this != &other)
        {
            delete descriptor;
            descriptor = other.descriptor;
            other.descriptor = nullptr;
        }
        return *this;
    }
    hipdnnBackendDescriptor_t get() const
    {
        return descriptor;
    }
    hipdnnBackendDescriptor_t* get_ptr()
    {
        return &descriptor;
    }
};
}
