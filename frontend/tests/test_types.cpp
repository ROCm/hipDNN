// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_frontend/types.hpp"
#include <gtest/gtest.h>

TEST(TestTypes, DataTypeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(to_sdk_type(DataType_t::FLOAT), hipdnn_sdk::data_objects::DataType::DataType_FLOAT);
    EXPECT_EQ(to_sdk_type(DataType_t::HALF), hipdnn_sdk::data_objects::DataType::DataType_HALF);
    EXPECT_EQ(to_sdk_type(DataType_t::BFLOAT16),
              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16);
    EXPECT_EQ(to_sdk_type(DataType_t::NOT_SET), hipdnn_sdk::data_objects::DataType::DataType_UNSET);
}

TEST(TestTypes, PointwiseModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(to_sdk_type(PointwiseMode_t::RELU_FWD),
              hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RELU_FWD);
    EXPECT_EQ(to_sdk_type(PointwiseMode_t::NOT_SET),
              hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_UNSET);
}

TEST(TestTypes, HeuristicModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(to_backend_type(HeurMode_t::FALLBACK),
              hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
}

TEST(TestTypes, GetDataTypeEnumFromType)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(get_data_type_enum_from_type<float>(), DataType_t::FLOAT);
    EXPECT_EQ(get_data_type_enum_from_type<half>(), DataType_t::HALF);
    EXPECT_EQ(get_data_type_enum_from_type<hip_bfloat16>(), DataType_t::BFLOAT16);
    EXPECT_EQ(get_data_type_enum_from_type<double>(), DataType_t::DOUBLE);

    EXPECT_EQ(get_data_type_enum_from_type<uint8_t>(), DataType_t::UINT8);
    EXPECT_EQ(get_data_type_enum_from_type<int32_t>(), DataType_t::INT32);

    // Test with an unsupported type (e.g., int)
    EXPECT_EQ(get_data_type_enum_from_type<int64_t>(), DataType_t::NOT_SET);
}
