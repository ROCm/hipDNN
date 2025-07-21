// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_utils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

using namespace miopen_legacy_plugin;

TEST(MiopenUtilsTest, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{.uid = 42, .ptr = reinterpret_cast<void*>(0x1234)},
           {.uid = 99, .ptr = reinterpret_cast<void*>(0x5678)}};

    auto result = miopen_utils::find_device_buffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(MiopenUtilsTest, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{.uid = 1, .ptr = reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        miopen_utils::find_device_buffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin::Hipdnn_plugin_exception);
}

TEST(MiopenUtilsTest, TensorDataTypeToMiopenDataType)
{
    using hipdnn_sdk::data_objects::DataType_BFLOAT16;
    using hipdnn_sdk::data_objects::DataType_FLOAT;
    using hipdnn_sdk::data_objects::DataType_HALF;

    EXPECT_EQ(miopen_utils::tensor_data_type_to_miopen_data_type(DataType_FLOAT), miopenFloat);
    EXPECT_EQ(miopen_utils::tensor_data_type_to_miopen_data_type(DataType_HALF), miopenHalf);
    EXPECT_EQ(miopen_utils::tensor_data_type_to_miopen_data_type(DataType_BFLOAT16),
              miopenBFloat16);
}

TEST(MiopenUtilsTest, TensorDataTypeToMiopenDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(miopen_utils::tensor_data_type_to_miopen_data_type(
                     static_cast<hipdnn_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin::Hipdnn_plugin_exception);
}
