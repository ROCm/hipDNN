// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
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
            throw hipdnn_plugin::HipdnnPluginException(                                 \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "MIOpen error occurred: " + std::string(miopenGetErrorString(status))); \
        }                                                                               \
    } while(0)

namespace miopen_legacy_plugin
{

namespace miopen_utils
{

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

miopenDataType_t tensorDataTypeToMiopenDataType(const hipdnn_sdk::data_objects::DataType& dataType);

}

}
