// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(HelpersTests, TryCatchSuccess)
{
    auto successFunction = []() -> hipdnnStatus_t { return HIPDNN_STATUS_SUCCESS; };

    hipdnnStatus_t status = tryCatch(successFunction);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HelpersTests, TryCatchException)
{
    auto exceptionFunction = []() -> hipdnnStatus_t { throw std::runtime_error("Test exception"); };

    hipdnnStatus_t status = tryCatch(exceptionFunction);
    EXPECT_EQ(status, HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(HelpersTests, TryCatchUnknownException)
{
    auto unknownExceptionFunction = []() -> hipdnnStatus_t {
        throw 42; // Throwing an unknown exception
    };

    hipdnnStatus_t status = tryCatch(unknownExceptionFunction);
    EXPECT_EQ(status, HIPDNN_STATUS_INTERNAL_ERROR);
}
