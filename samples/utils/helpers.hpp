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