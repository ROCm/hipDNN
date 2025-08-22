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

// NOLINTBEGIN(portability-template-virtual-member-function)

template <typename T>
class Migratable_memory_interface
{
public:
    virtual ~Migratable_memory_interface() = default;

    virtual T* host_data() = 0;
    virtual T* host_data_async() = 0;
    virtual const T* host_data() const = 0;
    virtual const T* host_data_async() const = 0;
    virtual void* device_data() = 0;
    virtual void* device_data_async() = 0;

    virtual void mark_host_modified() = 0;
    virtual void mark_device_modified() = 0;

    virtual size_t count() const = 0;
    virtual bool empty() const = 0;
    virtual Memory_location location() const = 0;

    virtual void resize(size_t new_count) = 0;
    virtual void clear() = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

template <class T, class HostAlloc = Host_allocator<T>, class DeviceAlloc = Device_allocator<T>>
class Migratable_memory : public Migratable_memory_interface<T>
{
    static_assert(std::is_base_of_v<Host_allocator_interface<T>, HostAlloc>,
                  "HostAlloc must derive from Host_allocator_interface<T>");
    static_assert(std::is_base_of_v<Device_allocator_interface<T>, DeviceAlloc>,
                  "DeviceAlloc must derive from Device_allocator_interface<T>");

public:
    explicit Migratable_memory(size_t count = 0, hipStream_t stream = nullptr)
        : _count(count)
        , _item_size(sizeof(T))
        , _total_size(count * _item_size)
        , _stream(stream)
    {
        if(count > 0)
        {
            allocate_host();
        }
    }

    ~Migratable_memory() override
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
        , _stream(other._stream)
        , _host_allocator(std::move(other._host_allocator))
        , _device_allocator(std::move(other._device_allocator))
    {
        other._host_ptr = nullptr;
        other._device_ptr = nullptr;
        other._count = 0;
        other._item_size = 0;
        other._total_size = 0;
        other._current_location = Memory_location::NONE;
        other._stream = nullptr;
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
            _stream = other._stream;
            _device_allocator = std::move(other._device_allocator);
            _host_allocator = std::move(other._host_allocator);

            other._host_ptr = nullptr;
            other._device_ptr = nullptr;
            other._count = 0;
            other._item_size = 0;
            other._total_size = 0;
            other._current_location = Memory_location::NONE;
            other._stream = nullptr;
            other._host_valid = false;
            other._device_valid = false;
        }
        return *this;
    }

    Migratable_memory(const Migratable_memory&) = delete;
    Migratable_memory& operator=(const Migratable_memory&) = delete;

    void resize(size_t new_count) override
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
    T* host_data() override
    {
        ensure_host_valid();
        return static_cast<T*>(_host_ptr);
    }

    T* host_data_async() override
    {
        ensure_host_valid(true);
        return static_cast<T*>(_host_ptr);
    }

    // Get host pointer (migrates if needed)
    const T* host_data() const override
    {
        ensure_const_host_valid();
        return static_cast<T*>(_host_ptr);
    }

    const T* host_data_async() const override
    {
        ensure_const_host_valid();
        return static_cast<T*>(_host_ptr);
    }

    void* device_data() override
    {
        ensure_device_valid();
        return static_cast<T*>(_device_ptr);
    }

    void* device_data_async() override
    {
        ensure_device_valid(true);
        return static_cast<T*>(_device_ptr);
    }

    // Mark memory as modified on host
    void mark_host_modified() override
    {
        _host_valid = true;
        _device_valid = false;
        _current_location = Memory_location::HOST;
    }

    // Mark memory as modified on device
    void mark_device_modified() override
    {
        _device_valid = true;
        _host_valid = false;
        _current_location = Memory_location::DEVICE;
    }

    size_t count() const override
    {
        return _count;
    }

    bool empty() const override
    {
        return _count == 0;
    }

    Memory_location location() const override
    {
        return _current_location;
    }

    void clear() override
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

    void ensure_const_host_valid() const
    {
        if((_host_ptr == nullptr) && _count > 0)
        {
            throw std::runtime_error("Host memory not allocated.");
        }

        if(!_host_valid && _device_valid && (_device_ptr != nullptr))
        {
            throw std::runtime_error(
                "Host memory is out of date and requires non-const access to update.");
        }
    }

    void ensure_host_valid(bool async = false)
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
                        _host_ptr, _device_ptr, _total_size, hipMemcpyDeviceToHost, _stream),
                    "Failed to copy from device to host");
            }
            else
            {
                throw_on_error(
                    hipMemcpyWithStream(
                        _host_ptr, _device_ptr, _total_size, hipMemcpyDeviceToHost, _stream),
                    "Failed to copy from device to host");
            }
            _host_valid = true;
            _current_location = Memory_location::BOTH;
        }
    }

    void ensure_device_valid(bool async = false)
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
                        _device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice, _stream),
                    "Failed to copy from host to device");
            }
            else
            {
                throw_on_error(
                    hipMemcpyWithStream(
                        _device_ptr, _host_ptr, _total_size, hipMemcpyHostToDevice, _stream),
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
    hipStream_t _stream{nullptr};
    HostAlloc _host_allocator;
    DeviceAlloc _device_allocator;
};

} // namespace utilities
} // namespace hipdnn_sdk
