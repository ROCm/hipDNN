// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(UtilitiesTests, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(UtilitiesTests, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}
