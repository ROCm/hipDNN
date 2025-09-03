// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

using namespace miopen_legacy_plugin;

TEST(TestMiopenUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{.uid = 42, .ptr = reinterpret_cast<void*>(0x1234)},
           {.uid = 99, .ptr = reinterpret_cast<void*>(0x5678)}};

    auto result = miopen_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestMiopenUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{.uid = 1, .ptr = reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        miopen_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin::HipdnnPluginException);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataType)
{
    using hipdnn_sdk::data_objects::DataType_BFLOAT16;
    using hipdnn_sdk::data_objects::DataType_FLOAT;
    using hipdnn_sdk::data_objects::DataType_HALF;

    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType_FLOAT), miopenFloat);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType_HALF), miopenHalf);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType_BFLOAT16), miopenBFloat16);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(miopen_utils::tensorDataTypeToMiopenDataType(
                     static_cast<hipdnn_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin::HipdnnPluginException);
}
