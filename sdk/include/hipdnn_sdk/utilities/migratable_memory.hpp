// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>

namespace hipdnn_sdk {
namespace utilities {

template<typename T>
class Migratable_memory {
public:

    enum class Location {
        HOST,
        DEVICE,
        BOTH,
        NONE
    };

    explicit Migratable_memory(size_t count = 0) 
        : _host_ptr(nullptr), _device_ptr(nullptr), _size(count), 
          _current_location(Location::NONE), _host_valid(false), _device_valid(false) {
        if (count > 0) {
            allocate_host();
        }
    }

    ~Migratable_memory() {
        cleanup();
    }

    Migratable_memory(Migratable_memory&& other) noexcept
        : _host_ptr(other._host_ptr), _device_ptr(other._device_ptr), 
          _size(other._size), _current_location(other._current_location),
          _host_valid(other._host_valid), _device_valid(other._device_valid) {
        other._host_ptr = nullptr;
        other._device_ptr = nullptr;
        other._size = 0;
        other._current_location = Location::NONE;
        other._host_valid = false;
        other._device_valid = false;
    }

    Migratable_memory& operator=(Migratable_memory&& other) noexcept {
        if (this != &other) {
            cleanup();
            _host_ptr = other._host_ptr;
            _device_ptr = other._device_ptr;
            _size = other._size;
            _current_location = other._current_location;
            _host_valid = other._host_valid;
            _device_valid = other._device_valid;
            
            other._host_ptr = nullptr;
            other._device_ptr = nullptr;
            other._size = 0;
            other._current_location = Location::NONE;
            other._host_valid = false;
            other._device_valid = false;
        }
        return *this;
    }

    Migratable_memory(const Migratable_memory&) = delete;
    Migratable_memory& operator=(const Migratable_memory&) = delete;

    void resize(size_t new_count) {
        cleanup();
        _size = new_count;
        _current_location = Location::NONE;
        _host_valid = false;
        _device_valid = false;
        if (new_count > 0) {
            allocate_host();
        }
    }

    // Get host pointer (migrates if needed)
    T* host_data() {
        ensure_host_valid();
        return _host_ptr;
    }

    // Get device pointer (migrates if needed)
    T* device_data() {
        ensure_device_valid();
        return _device_ptr;
    }

    // Get const host pointer (migrates if needed)
    const T* host_data() const {
        const_cast<Migratable_memory*>(this)->ensure_host_valid();
        return _host_ptr;
    }

    // Get const device pointer (migrates if needed)
    const T* device_data() const {
        const_cast<Migratable_memory*>(this)->ensure_device_valid();
        return _device_ptr;
    }

    // Mark memory as modified on host
    void mark_host_modified() {
        _host_valid = true;
        _device_valid = false;
        _current_location = Location::HOST;
    }

    // Mark memory as modified on device
    void mark_device_modified() {
        _device_valid = true;
        _host_valid = false;
        _current_location = Location::DEVICE;
    }

    size_t size() const { return _size; }

    bool empty() const { return _size == 0; }

    Location location() const { return _current_location; }

    void clear() {
        cleanup();
        _size = 0;
        _current_location = Location::NONE;
        _host_valid = false;
        _device_valid = false;
    }   

private:
    void throw_on_error(hipError_t err, const char* msg) {
        if (err != hipSuccess) {
            throw std::runtime_error(msg);
        }
    }

    // TODO - Consider different allocation strategies, such as unified memory, host pinned memory, etc.
    // For now, we will use hipHostMalloc for host memory and hipMalloc for device
    // memory. This can be extended based on specific requirements.

    void allocate_host() {
        if (!_host_ptr && _size > 0) {
            throw_on_error(hipHostMalloc(&_host_ptr, _size * sizeof(T)), "Failed to allocate host memory");
            _host_valid = true;
            _current_location = Location::HOST;
        }
    }

    void allocate_device() {
        if (!_device_ptr && _size > 0) {
            throw_on_error(hipMalloc(&_device_ptr, _size * sizeof(T)), "Failed to allocate device memory");
        }
    }

    void ensure_host_valid() {
        if (_size == 0) return;
        
        allocate_host();
        
        if (!_host_valid && _device_valid && _device_ptr) {
            throw_on_error(hipMemcpy(_host_ptr, _device_ptr, _size * sizeof(T), hipMemcpyDeviceToHost), "Failed to copy from device to host");
            _host_valid = true;
            _current_location = Location::BOTH;
        }
    }

    void ensure_device_valid() {
        if (_size == 0) return;
        
        allocate_device();
        
        if (!_device_valid && _host_valid && _host_ptr) {
            throw_on_error(hipMemcpy(_device_ptr, _host_ptr, _size * sizeof(T), hipMemcpyHostToDevice), "Failed to copy from host to device");
            _device_valid = true;
            _current_location = Location::BOTH;
        }
    }

    void cleanup() {
        if (_host_ptr) {
            throw_on_error(hipHostFree(_host_ptr), "Failed to free host memory");
            _host_ptr = nullptr;
        }
        if (_device_ptr) {
            throw_on_error(hipFree(_device_ptr), "Failed to free device memory");
            _device_ptr = nullptr;
        }
        _host_valid = false;
        _device_valid = false;
        _current_location = Location::NONE;
    }

    T* _host_ptr;
    T* _device_ptr;
    size_t _size;
    Location _current_location;
    bool _host_valid;
    bool _device_valid;
};

} // namespace utilities
} // namespace hipdnn_sdk
