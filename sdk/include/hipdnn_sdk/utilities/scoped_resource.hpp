// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Scoped_resource is a utility class that manages a resource with a custom destructor.

#include <utility>

namespace hipdnn::sdk::utilities
{

template <typename T, typename Destructor>
class Scoped_resource
{
public:
    Scoped_resource() = default;
    Scoped_resource(T resource, Destructor destructor)
        : _resource(resource)
        , _destructor(destructor)
        , _empty(false)
    {
    }

    ~Scoped_resource()
    {
        if(!_empty)
            _destructor(_resource);
    }

    // Prevent copying
    Scoped_resource(const Scoped_resource&) = delete;
    Scoped_resource& operator=(const Scoped_resource&) = delete;

    // Allow moving
    Scoped_resource(Scoped_resource&& other)
    {
        if(other._empty)
            return;

        _resource = std::move(other._resource);
        _destructor = std::move(other._destructor);
        other._empty = true;
        _empty = false;
    }

    Scoped_resource& operator=(Scoped_resource&& other)
    {
        if(this != &other)
        {
            if(!_empty)
                _destructor(_resource);

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

    bool is_empty() const
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
