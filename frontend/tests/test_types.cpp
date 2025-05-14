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
