// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_type_helpers.hpp>
#include <miopen/miopen.h>

#define LOG_ON_MIOPEN_FAILURE(status)                                                    \
    do                                                                                   \
    {                                                                                    \
        if(status != miopenStatusSuccess)                                                \
        {                                                                                \
            HIPDNN_LOG_ERROR("MIOpen error occurred: {}", miopenGetErrorString(status)); \
        }                                                                                \
    } while(0)

#define THROW_ON_MIOPEN_FAILURE(status)                                                 \
    do                                                                                  \
    {                                                                                   \
        if(status != miopenStatusSuccess)                                               \
        {                                                                               \
            throw hipdnn_plugin::Hipdnn_plugin_exception(                               \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "MIOpen error occurred: " + std::string(miopenGetErrorString(status))); \
        }                                                                               \
    } while(0)

namespace miopen_legacy_plugin
{

namespace miopen_utils
{

hipdnnPluginDeviceBuffer_t find_device_buffer(int64_t uid,
                                              const hipdnnPluginDeviceBuffer_t* device_buffers,
                                              uint32_t num_device_buffers);

miopenDataType_t
    tensor_data_type_to_miopen_data_type(const hipdnn_sdk::data_objects::DataType& data_type);

}

}