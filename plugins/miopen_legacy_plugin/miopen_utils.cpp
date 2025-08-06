// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_utils.hpp"

namespace miopen_legacy_plugin
{

namespace miopen_utils
{

hipdnnPluginDeviceBuffer_t find_device_buffer(int64_t uid,
                                              const hipdnnPluginDeviceBuffer_t* device_buffers,
                                              uint32_t num_device_buffers)
{
    for(uint32_t i = 0; i < num_device_buffers; i++)
    {
        if(uid == device_buffers[i].uid)
        {
            return device_buffers[i];
        }
    }

    throw hipdnn_plugin::Hipdnn_plugin_exception(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "Device buffer with the uid: " + std::to_string(uid)
            + " not found in the provided device buffers.");
}

miopenDataType_t
    tensor_data_type_to_miopen_data_type(const hipdnn_sdk::data_objects::DataType& data_type)
{
    switch(data_type)
    {
    case hipdnn_sdk::data_objects::DataType_FLOAT:
        return miopenFloat;
    case hipdnn_sdk::data_objects::DataType_HALF:
        return miopenHalf;
    case hipdnn_sdk::data_objects::DataType_BFLOAT16:
        return miopenBFloat16;
    default:
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for MIOpen: "
                + std::string(hipdnn_sdk::data_objects::to_string(data_type)));
    }
}

}

}
