// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_frontend/types.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(TestTypes, DataTypeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(to_sdk_type(DataType_t::FLOAT), hipdnn_sdk::data_objects::DataType::DataType_FLOAT);
    EXPECT_EQ(to_sdk_type(DataType_t::HALF), hipdnn_sdk::data_objects::DataType::DataType_HALF);
    EXPECT_EQ(to_sdk_type(DataType_t::BFLOAT16),
              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16);
    EXPECT_EQ(to_sdk_type(DataType_t::DOUBLE), hipdnn_sdk::data_objects::DataType::DataType_DOUBLE);
    EXPECT_EQ(to_sdk_type(DataType_t::UINT8), hipdnn_sdk::data_objects::DataType::DataType_UINT8);
    EXPECT_EQ(to_sdk_type(DataType_t::INT32), hipdnn_sdk::data_objects::DataType::DataType_INT32);
    EXPECT_EQ(to_sdk_type(DataType_t::NOT_SET), hipdnn_sdk::data_objects::DataType::DataType_UNSET);
}

TEST(TestTypes, ConvolutionModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(to_sdk_type(ConvolutionMode_t::CROSS_CORRELATION),
              hipdnn_sdk::data_objects::ConvMode::ConvMode_CROSS_CORRELATION);
    EXPECT_EQ(to_sdk_type(ConvolutionMode_t::CONVOLUTION),
              hipdnn_sdk::data_objects::ConvMode::ConvMode_CONVOLUTION);
    EXPECT_EQ(to_sdk_type(ConvolutionMode_t::NOT_SET),
              hipdnn_sdk::data_objects::ConvMode::ConvMode_UNSET);
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

    EXPECT_EQ(get_data_type_enum_from_type<float*>(), DataType_t::NOT_SET);
    EXPECT_EQ(get_data_type_enum_from_type<char>(), DataType_t::NOT_SET);
}

TEST(TestTypes, DataTypeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(DataType_t::FLOAT), "fp32");
    EXPECT_STREQ(to_string(DataType_t::HALF), "fp16");
    EXPECT_STREQ(to_string(DataType_t::BFLOAT16), "bf16");
    EXPECT_STREQ(to_string(DataType_t::DOUBLE), "fp64");
    EXPECT_STREQ(to_string(DataType_t::UINT8), "uint8");
    EXPECT_STREQ(to_string(DataType_t::INT32), "int32");
    EXPECT_STREQ(to_string(DataType_t::NOT_SET), "unknown");
}

TEST(TestTypes, DataTypeStreamOperator)
{
    using namespace hipdnn_frontend;

    std::ostringstream oss;

    oss << DataType_t::FLOAT;
    EXPECT_EQ(oss.str(), "fp32");
    oss.str("");

    oss << DataType_t::HALF;
    EXPECT_EQ(oss.str(), "fp16");
    oss.str("");

    oss << DataType_t::BFLOAT16;
    EXPECT_EQ(oss.str(), "bf16");
    oss.str("");

    oss << DataType_t::DOUBLE;
    EXPECT_EQ(oss.str(), "fp64");
    oss.str("");

    oss << DataType_t::UINT8;
    EXPECT_EQ(oss.str(), "uint8");
    oss.str("");

    oss << DataType_t::INT32;
    EXPECT_EQ(oss.str(), "int32");
    oss.str("");

    oss << DataType_t::NOT_SET;
    EXPECT_EQ(oss.str(), "unknown");
}
