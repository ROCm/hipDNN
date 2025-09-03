/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

#define HIPDNN_FRONTEND_TESTS "hipdnn_frontend_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_sdk::test_utilities::initializeSpdlogDefaultLogger(HIPDNN_FRONTEND_TESTS);

    return RUN_ALL_TESTS();
}
