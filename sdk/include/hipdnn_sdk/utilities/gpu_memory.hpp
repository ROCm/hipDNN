/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
*/
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include <numeric>
#include <stdexcept>

#include "hipdnn_sdk/logging/logger.hpp"

namespace hipdnn::sdk::utilities
{

template <typename Tgpu>
class Gpu_memory
{
public:
    explicit Gpu_memory(std::size_t num_elements)
        : _num_elements(num_elements)
        , _device_ptr(nullptr)
    {
        HIPDNN_LOG_INFO("Allocating GPU memory of size "
                        + std::to_string(_num_elements * sizeof(Tgpu)) + " bytes.");
        if(hipMalloc(reinterpret_cast<void**>(&_device_ptr), _num_elements * sizeof(Tgpu))
           != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed to allocate GPU memory of size "
                                     + std::to_string(_num_elements * sizeof(Tgpu)));
        }
    }

    explicit Gpu_memory(const std::vector<int64_t>& dims)
        : Gpu_memory(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>()))
    {
    }

    ~Gpu_memory()
    {
        if(_device_ptr)
        {
            if(hipFree(_device_ptr) != hipSuccess)
            {
                HIPDNN_LOG_ERROR("hipFree failed to free GPU memory.");
            }
        }
    }

    Tgpu* data()
    {
        return _device_ptr;
    }

    const Tgpu* data() const
    {
        return _device_ptr;
    }

    std::size_t size() const
    {
        return _num_elements;
    }

private:
    std::size_t _num_elements;
    Tgpu* _device_ptr;
};

}