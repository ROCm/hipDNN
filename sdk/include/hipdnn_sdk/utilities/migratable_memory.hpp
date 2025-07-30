// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <memory>
#include <stdexcept>

namespace hipdnn_sdk
{
namespace utilities
{

/// @brief A class that manages memory that can be migrated between host and device.
/// It provides functionality to allocate, resize, and access memory on both host and device,
/// while ensuring that data is synchronized as needed.  This class is not thread safe.
///
class Migratable_memory
{
public:
    enum class Location
    {
        HOST,
        DEVICE,
        BOTH,
        NONE
    };

    explicit Migratable_memory(size_t count = 0, size_t item_size = 0)
        : _count(count)
        , _item_size(item_size)
        , _total_size(count * item_size)
    {
        if(count > 0)
        {
            allocate_host();
        }
    }

    ~Migratable_memory()
    {
        cleanup();
    }

    Migratable_memory(Migratable_memory&& other) noexcept
        : _host_ptr(other._host_ptr)
        , _device_ptr(other._device_ptr)
        , _count(other._count)
        , _item_size(other._item_size)
        , _total_size(other._total_size)
        , _current_location(other._current_location)
        , _host_valid(other._host_valid)
        , _device_valid(other._device_valid)
    {
        other._host_ptr = nullptr;
        other._device_ptr = nullptr;
        other._count = 0;
        other._item_size = 0;
        other._total_size = 0;
        other._current_location = Location::NONE;
        other._host_valid = false;
        other._device_valid = false;
    }

    Migratable_memory& operator=(Migratable_memory&& other) noexcept
    {
        if(this != &other)
        {
            cleanup();
            _host_ptr = other._host_ptr;
            _device_ptr = other._device_ptr;
            _count = other._count;
            _item_size = other._item_size;
            _total_size = other._total_size;
            _current_location = other._current_location;
            _host_valid = other._host_valid;
            _device_valid = other._device_valid;

            other._host_ptr = nullptr;
            other._device_ptr = nullptr;
            other._count = 0;
            other._item_size = 0;
            other._total_size = 0;
            other._current_location = Location::NONE;
            other._host_valid = false;
            other._device_valid = false;
        }
        return *this;
    }

    Migratable_memory(const Migratable_memory&) = delete;
    Migratable_memory& operator=(const Migratable_memory&) = delete;

    void resize(size_t new_count)
    {
        cleanup();
        _count = new_count;
        _total_size = new_count * _item_size;
        _current_location = Location::NONE;
        _host_valid = false;
        _device_valid = false;
        if(new_count > 0)
        {
            allocate_host();
        }
    }

    // Get host pointer (migrates if needed)
    template <typename T>
    T* host_data()
    {
        ensure_host_valid();
        return static_cast<T*>(_host_ptr);
    }

    // Get device pointer (migrates if needed)
    template <typename T>
    T* device_data()
    {
        ensure_device_valid();
        return static_cast<T*>(_device_ptr);
    }

    // Get device pointer (migrates if needed)
    template <typename T>
    T* device_data(hipStream_t stream)
    {
        ensure_device_valid(stream);
        return static_cast<T*>(_device_ptr);
    }

    // Get const host pointer (migrates if needed)
    template <typename T>
    const T* host_data() const
    {
        const_cast<Migratable_memory*>(this)->ensure_host_valid();
        return static_cast<T*>(_host_ptr);
    }

    // Get const device pointer (migrates if needed)
    template <typename T>
    const T* device_data() const
    {
        const_cast<Migratable_memory*>(this)->ensure_device_valid();
        return static_cast<T*>(_device_ptr);
    }

    // Get const device pointer (migrates if needed)
    template <typename T>
    const T* device_data(hipStream_t stream) const
    {
        const_cast<Migratable_memory*>(this)->ensure_device_valid(stream);
        return static_cast<T*>(_device_ptr);
    }

    // Mark memory as modified on host
    void mark_host_modified()
    {
        _host_valid = true;
        _device_valid = false;
        _current_location = Location::HOST;
    }

    // Mark memory as modified on device
    void mark_device_modified()
    {
        _device_valid = true;
        _host_valid = false;
        _current_location = Location::DEVICE;
    }

    size_t count() const
    {
        return _count;
    }

    bool empty() const
    {
        return _count == 0;
    }

    Location location() const
    {
        return _current_location;
    }

    void clear()
    {
        cleanup();
        _count = 0;
        _item_size = 0;
        _total_size = 0;
        _current_location = Location::NONE;
        _host_valid = false;
        _device_valid = false;
    }

private:
    static void throw_on_error(hipError_t err, const char* msg)
    {
        if(err != hipSuccess)
        {
            throw std::runtime_error(msg);
        }
    }

    static void log_on_error(hipError_t err, const char* msg)
    {
        std::ignore = msg;

        if(err != hipSuccess)
        {
            HIPDNN_LOG_ERROR("{}: HIP error: {}", msg, hipGetErrorString(err));
        }
    }

    // TODO - Consider different allocation strategies, such as unified memory, host pinned memory, etc.
    // For now, we will use hipHostMalloc for host memory and hipMalloc for device
    // memory. This can be extended based on specific requirements.

    void allocate_host()
    {
        if((_host_ptr == nullptr) && _total_size > 0)
        {
            throw_on_error(hipHostMalloc(&_host_ptr, _total_size),
                           "Failed to allocate host memory");
            _host_valid = true;
            _current_location = Location::HOST;
        }
    }

    void allocate_device()
    {
        if((_device_ptr == nullptr) && _total_size > 0)
        {
            throw_on_error(hipMalloc(&_device_ptr, _total_size),
                           "Failed to allocate device memory");
        }
    }

    void ensure_host_valid()
    {
        if(_count == 0)
        {
            return;
        }

        allocate_host();

        if(!_host_valid && _device_valid && (_device_ptr != nullptr))
        {
            throw_on_error(hipMemcpy(_host_ptr, _device_ptr, _total_size, hipMemcpyDeviceToHost),
                           "Failed to copy from device to host");
            _host_valid = true;
            _current_location = Location::BOTH;
        }
    }

    void ensure_device_valid()
    {
        if(_count == 0)
        {
            return;
        }

        allocate_device();

        if(!_device_valid && _host_valid && (_host_ptr != nullptr))
        {
            throw_on_error(hipMemcpy(_device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice),
                           "Failed to copy from host to device");
            _device_valid = true;
            _current_location = Location::BOTH;
        }
    }

    void ensure_device_valid(hipStream_t stream)
    {
        if(_count == 0)
        {
            return;
        }

        allocate_device();

        if(!_device_valid && _host_valid && (_host_ptr != nullptr))
        {
            throw_on_error(hipMemcpyWithStream(
                               _device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice, stream),
                           "Failed to copy from host to device");
            _device_valid = true;
            _current_location = Location::BOTH;
        }
    }

    void cleanup()
    {
        if(_host_ptr != nullptr)
        {
            log_on_error(hipHostFree(_host_ptr), "Failed to free host memory");
            _host_ptr = nullptr;
        }
        if(_device_ptr != nullptr)
        {
            log_on_error(hipFree(_device_ptr), "Failed to free device memory");
            _device_ptr = nullptr;
        }
        _host_valid = false;
        _device_valid = false;
        _current_location = Location::NONE;
    }

    void* _host_ptr{nullptr};
    void* _device_ptr{nullptr};
    size_t _count;
    size_t _item_size;
    size_t _total_size;
    Location _current_location{Location::NONE};
    bool _host_valid{false};
    bool _device_valid{false};
};

} // namespace utilities
} // namespace hipdnn_sdk
