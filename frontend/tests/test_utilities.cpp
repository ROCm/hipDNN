// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(UtilitiesTests, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(common_shape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(UtilitiesTests, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> input_shapes = {};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(common_shape, (std::vector<int64_t>{1, 2, 3}));
}