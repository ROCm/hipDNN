/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

#define MIOPEN_FRONTEND_INTEGRATION_TESTS "miopen_frontend_integration_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    logging_test_utils::initializeSpdlogDefaultLogger(MIOPEN_FRONTEND_INTEGRATION_TESTS);

    return RUN_ALL_TESTS();
}
