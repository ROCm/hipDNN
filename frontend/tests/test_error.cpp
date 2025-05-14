// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/error.hpp>

TEST(ErrorTests, DefaultConstructor)
{
    hipdnn_frontend::error_t error;
    EXPECT_EQ(error.get_code(), hipdnn_frontend::error_code_t::OK);
    EXPECT_TRUE(error.is_good());
    EXPECT_FALSE(error.is_bad());
    EXPECT_EQ(error.get_message(), "");
}

TEST(ErrorTests, ParameterizedConstructor)
{
    hipdnn_frontend::error_t error(hipdnn_frontend::error_code_t::INVALID_VALUE,
                                   "Invalid value provided");
    EXPECT_EQ(error.get_code(), hipdnn_frontend::error_code_t::INVALID_VALUE);
    EXPECT_FALSE(error.is_good());
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_message(), "Invalid value provided");
}

TEST(ErrorTests, EqualityOperators)
{
    hipdnn_frontend::error_t error1(hipdnn_frontend::error_code_t::INVALID_VALUE, "Error 1");
    hipdnn_frontend::error_t error2(hipdnn_frontend::error_code_t::INVALID_VALUE, "Error 2");
    hipdnn_frontend::error_t error3(hipdnn_frontend::error_code_t::ATTRIBUTE_NOT_SET, "Error 3");

    EXPECT_TRUE(error1 == error2);
    EXPECT_FALSE(error1 == error3);
    EXPECT_TRUE(error1 != error3);
    EXPECT_FALSE(error1 != error2);
}

TEST(ErrorTests, CodeEqualityOperators)
{
    hipdnn_frontend::error_t error(hipdnn_frontend::error_code_t::INVALID_VALUE,
                                   "Invalid value provided");

    EXPECT_TRUE(error == hipdnn_frontend::error_code_t::INVALID_VALUE);
    EXPECT_FALSE(error == hipdnn_frontend::error_code_t::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error != hipdnn_frontend::error_code_t::ATTRIBUTE_NOT_SET);
    EXPECT_FALSE(error != hipdnn_frontend::error_code_t::INVALID_VALUE);
}

TEST(ErrorTests, CheckHipdnnErrorMacro)
{
    auto success_function = []() -> hipdnn_frontend::error_t {
        return {hipdnn_frontend::error_code_t::OK, "Success"};
    };

    auto failure_function = []() -> hipdnn_frontend::error_t {
        return {hipdnn_frontend::error_code_t::INVALID_VALUE, "Failure"};
    };

    auto test_function = [&]() -> hipdnn_frontend::error_t {
        CHECK_HIPDNN_ERROR(success_function());
        CHECK_HIPDNN_ERROR(failure_function());
        return {hipdnn_frontend::error_code_t::OK, "Should not reach here"};
    };

    hipdnn_frontend::error_t result = test_function();
    EXPECT_EQ(result.get_code(), hipdnn_frontend::error_code_t::INVALID_VALUE);
    EXPECT_EQ(result.get_message(), "Failure");
}