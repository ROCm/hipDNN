// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// ScopedResource is a utility class that manages a resource with a custom destructor.

#include <utility>

namespace hipdnn::sdk::utilities
{

template <typename T, typename Destructor>
class ScopedResource
{
public:
    ScopedResource() = default;
    ScopedResource(T resource, Destructor destructor)
        : _resource(resource)
        , _destructor(destructor)
        , _empty(false)
    {
    }

    ~ScopedResource()
    {
        if(!_empty)
        {
            _destructor(_resource);
        }
    }

    // Prevent copying
    ScopedResource(const ScopedResource&) = delete;
    ScopedResource& operator=(const ScopedResource&) = delete;

    // Allow moving
    ScopedResource(ScopedResource&& other) noexcept
    {
        if(other._empty)
        {
            return;
        }

        _resource = std::move(other._resource);
        _destructor = std::move(other._destructor);
        other._empty = true;
        _empty = false;
    }

    ScopedResource& operator=(ScopedResource&& other) noexcept
    {
        if(this != &other)
        {
            if(!_empty)
            {
                _destructor(_resource);
            }

            _empty = other._empty;
            if(!other._empty)
            {
                _resource = std::move(other._resource);
                _destructor = std::move(other._destructor);
                other._empty = true;
            }
        }
        return *this;
    }

    bool isEmpty() const
    {
        return _empty;
    }

    T get() const
    {
        return _resource;
    }

private:
    T _resource;
    Destructor _destructor;
    bool _empty = true;
};

} // namespace hipdnn::sdk::utilities
