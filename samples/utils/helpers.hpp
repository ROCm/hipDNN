// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

#define HIPDNN_CHECK(status)                                                             \
    do                                                                                   \
    {                                                                                    \
        if(status != HIPDNN_STATUS_SUCCESS)                                              \
        {                                                                                \
            std::cerr << "MIOpen Error: " << hipdnnGetErrorString(status) << " in file " \
                      << __FILE__ << " at line " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while(0)

#define HIPDNN_FE_CHECK(status_obj)                                                       \
    do                                                                                    \
    {                                                                                     \
        auto const& status = status_obj;                                                  \
        if(!status.is_good())                                                             \
        {                                                                                 \
            std::cerr << "hipDNN Frontend Error: " << status.get_message() << " in file " \
                      << __FILE__ << " at line " << __LINE__ << std::endl;                \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while(0)

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    create_tensor(const std::vector<int64_t>& dims, hipdnn_frontend::DataType_t data_type)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(dims).set_data_type(data_type);

    std::vector<int64_t> strides(dims.size());
    if(!dims.empty())
    {
        strides.back() = 1;
        for(int i = dims.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }
    tensor->set_stride(strides);
    return tensor;
}

inline int64_t get_tensor_element_count(
    const std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& tensor)
{
    int64_t count = 1;
    for(auto dim : tensor->get_dim())
    {
        count *= dim;
    }
    return count;
}

template <typename T>
void init_data(T* data, int64_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(int64_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(dis(gen));
    }
}

inline size_t get_data_type_size(hipdnn_frontend::DataType_t data_type)
{
    switch(data_type)
    {
    case hipdnn_frontend::DataType_t::FLOAT:
        return sizeof(float);
    case hipdnn_frontend::DataType_t::HALF:
        return sizeof(unsigned short);
    case hipdnn_frontend::DataType_t::BFLOAT16:
        return sizeof(unsigned short);
    default:
        return 0;
    }
}

inline int64_t
    get_tensor_size(const std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& tensor)
{
    int64_t size = 1;
    for(auto dim : tensor->get_dim())
    {
        size *= dim;
    }
    return size * sizeof(tensor->get_data_type());
}

template <typename T_ELEM>
class Surface
{
public:
    T_ELEM* devPtr = nullptr;
    T_ELEM* hostPtr = nullptr;
    int64_t n_elems = 0;

    explicit Surface(int64_t num_elements)
        : n_elems(num_elements)
    {
        allocate();
        init_data(hostPtr, n_elems);
        copy_to_device();
    }

    explicit Surface(int64_t num_elements, T_ELEM fill_value)
        : n_elems(num_elements)
    {
        allocate();
        for(int64_t i = 0; i < n_elems; i++)
        {
            hostPtr[i] = fill_value;
        }
        copy_to_device();
    }

    ~Surface()
    {
        if(devPtr)
        {
            HIP_CHECK(hipFree(devPtr));
            devPtr = nullptr;
        }
        if(hostPtr)
        {
            free(hostPtr);
            hostPtr = nullptr;
        }
    }

    Surface(const Surface& other)
        : n_elems(other.n_elems)
    {
        allocate();
        std::copy(other.hostPtr, other.hostPtr + n_elems, hostPtr);
        copy_to_device();
    }

    Surface& operator=(Surface other)
    {
        swap(*this, other);
        return *this;
    }

    Surface(Surface&& other) noexcept
        : Surface()
    {
        swap(*this, other);
    }

    friend void swap(Surface& first, Surface& second) noexcept
    {
        using std::swap;
        swap(first.n_elems, second.n_elems);
        swap(first.hostPtr, second.hostPtr);
        swap(first.devPtr, second.devPtr);
    }

    void copy_to_host()
    {
        HIP_CHECK(hipMemcpy(hostPtr, devPtr, n_elems * sizeof(T_ELEM), hipMemcpyDeviceToHost));
    }

protected:
    explicit Surface() {}

private:
    void allocate()
    {
        HIP_CHECK(hipMalloc(&devPtr, n_elems * sizeof(T_ELEM)));
        hostPtr = static_cast<T_ELEM*>(calloc(n_elems, sizeof(T_ELEM)));
        if(hostPtr == nullptr)
        {
            std::cerr << "Failed to allocate host memory." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void copy_to_device()
    {
        HIP_CHECK(hipMemcpy(devPtr, hostPtr, n_elems * sizeof(T_ELEM), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
    }
};