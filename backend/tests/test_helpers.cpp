// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "helpers.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(HelpersTests, TryCatchSuccess)
{
    auto success_function = []() -> hipdnnStatus_t { return HIPDNN_STATUS_SUCCESS; };

    hipdnnStatus_t status = try_catch(success_function);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HelpersTests, TryCatchException)
{
    auto exception_function
        = []() -> hipdnnStatus_t { throw std::runtime_error("Test exception"); };

    hipdnnStatus_t status = try_catch(exception_function);
    EXPECT_EQ(status, HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(HelpersTests, TryCatchUnknownException)
{
    auto unknown_exception_function = []() -> hipdnnStatus_t {
        throw 42; // Throwing an unknown exception
    };

    hipdnnStatus_t status = try_catch(unknown_exception_function);
    EXPECT_EQ(status, HIPDNN_STATUS_INTERNAL_ERROR);
}
