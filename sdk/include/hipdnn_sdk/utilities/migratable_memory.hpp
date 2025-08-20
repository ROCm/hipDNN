// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/utilities/allocators.hpp>
#include <memory>
#include <stdexcept>

namespace hipdnn_sdk
{
namespace utilities
{

enum class Memory_location
{
    HOST,
    DEVICE,
    BOTH,
    NONE
};

/// @brief A class that manages memory that can be migrated between host and device.
/// It provides functionality to allocate, resize, and access memory on both host and device,
/// while ensuring that data is synchronized as needed.  This class is not thread safe.
///
/// @tparam T The type of elements stored
/// @tparam HostAlloc The host allocator type (defaults to Host_allocator<T>)
/// @tparam DeviceAlloc The device allocator type (defaults to Device_allocator<T>)
template <class T, class HostAlloc = Host_allocator<T>, class DeviceAlloc = Device_allocator<T>>
class Migratable_memory
{
    static_assert(std::is_base_of_v<Host_allocator_interface<T>, HostAlloc>,
                  "HostAlloc must derive from Host_allocator_interface<T>");
    static_assert(std::is_base_of_v<Device_allocator_interface<T>, DeviceAlloc>,
                  "DeviceAlloc must derive from Device_allocator_interface<T>");

public:
    explicit Migratable_memory(size_t count = 0)
        : _count(count)
        , _item_size(sizeof(T))
        , _total_size(count * _item_size)
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
        other._current_location = Memory_location::NONE;
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
            other._current_location = Memory_location::NONE;
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
        _current_location = Memory_location::NONE;
        _host_valid = false;
        _device_valid = false;
        if(new_count > 0)
        {
            allocate_host();
        }
    }

    // Get host pointer (migrates if needed)
    T* host_data(hipStream_t stream = nullptr)
    {
        ensure_host_valid(stream);
        return static_cast<T*>(_host_ptr);
    }

    T* host_data_async(hipStream_t stream = nullptr)
    {
        ensure_host_valid(stream, true);
        return static_cast<T*>(_host_ptr);
    }

    const T* host_data(hipStream_t stream = nullptr) const
    {
        const_cast<Migratable_memory*>(this)->ensure_host_valid(stream);
        return static_cast<T*>(_host_ptr);
    }

    const T* host_data_async(hipStream_t stream = nullptr) const
    {
        const_cast<Migratable_memory*>(this)->ensure_host_valid(stream, true);
        return static_cast<T*>(_host_ptr);
    }

    void* device_data(hipStream_t stream = nullptr)
    {
        ensure_device_valid(stream);
        return static_cast<T*>(_device_ptr);
    }

    void* device_data_async(hipStream_t stream = nullptr)
    {
        ensure_device_valid(stream, true);
        return static_cast<T*>(_device_ptr);
    }

    void* device_data(hipStream_t stream = nullptr) const
    {
        const_cast<Migratable_memory*>(this)->ensure_device_valid(stream);
        return static_cast<T*>(_device_ptr);
    }

    void* device_data_async(hipStream_t stream = nullptr) const
    {
        const_cast<Migratable_memory*>(this)->ensure_device_valid(stream, true);
        return static_cast<T*>(_device_ptr);
    }

    // Mark memory as modified on host
    void mark_host_modified()
    {
        _host_valid = true;
        _device_valid = false;
        _current_location = Memory_location::HOST;
    }

    // Mark memory as modified on device
    void mark_device_modified()
    {
        _device_valid = true;
        _host_valid = false;
        _current_location = Memory_location::DEVICE;
    }

    size_t count() const
    {
        return _count;
    }

    bool empty() const
    {
        return _count == 0;
    }

    Memory_location location() const
    {
        return _current_location;
    }

    void clear()
    {
        cleanup();
        _count = 0;
        _item_size = 0;
        _total_size = 0;
        _current_location = Memory_location::NONE;
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

    void allocate_host()
    {
        if((_host_ptr == nullptr) && _count > 0)
        {
            _host_ptr = _host_allocator.allocate(_count);
            _host_valid = true;
            _current_location = Memory_location::HOST;
        }
    }

    void allocate_device()
    {
        if((_device_ptr == nullptr) && _count > 0)
        {
            _device_ptr = _device_allocator.allocate(_count);
        }
    }

    void ensure_host_valid(hipStream_t stream = nullptr, bool async = false)
    {
        if(_count == 0)
        {
            return;
        }

        allocate_host();

        if(!_host_valid && _device_valid && (_device_ptr != nullptr))
        {
            if(async)
            {
                throw_on_error(
                    hipMemcpyAsync(
                        _host_ptr, _device_ptr, _total_size, hipMemcpyDeviceToHost, stream),
                    "Failed to copy from device to host");
            }
            else
            {
                throw_on_error(
                    hipMemcpyWithStream(
                        _host_ptr, _device_ptr, _total_size, hipMemcpyDeviceToHost, stream),
                    "Failed to copy from device to host");
            }
            _host_valid = true;
            _current_location = Memory_location::BOTH;
        }
    }

    void ensure_device_valid(hipStream_t stream = nullptr, bool async = false)
    {
        if(_count == 0)
        {
            return;
        }

        allocate_device();

        if(!_device_valid && _host_valid && (_host_ptr != nullptr))
        {
            if(async)
            {
                throw_on_error(
                    hipMemcpyAsync(
                        _device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice, stream),
                    "Failed to copy from host to device");
            }
            else
            {
                throw_on_error(
                    hipMemcpyWithStream(
                        _device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice, stream),
                    "Failed to copy from host to device");
            }
            _device_valid = true;
            _current_location = Memory_location::BOTH;
        }
    }

    void cleanup()
    {
        if(_host_ptr != nullptr)
        {
            _host_allocator.deallocate(static_cast<T*>(_host_ptr), _count);
            _host_ptr = nullptr;
        }
        if(_device_ptr != nullptr)
        {
            _device_allocator.deallocate(static_cast<T*>(_device_ptr), _count);
            _device_ptr = nullptr;
        }
        _host_valid = false;
        _device_valid = false;
        _current_location = Memory_location::NONE;
    }

    void* _host_ptr{nullptr};
    void* _device_ptr{nullptr};
    size_t _count;
    size_t _item_size;
    size_t _total_size;
    Memory_location _current_location{Memory_location::NONE};
    bool _host_valid{false};
    bool _device_valid{false};
    HostAlloc _host_allocator;
    DeviceAlloc _device_allocator;
};

} // namespace utilities
} // namespace hipdnn_sdk
