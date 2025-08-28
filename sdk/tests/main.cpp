/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

#define HIPDNN_SDK_TESTS "hipdnn_sdk_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn::logging::initializeCallbackLogging(HIPDNN_SDK_TESTS,
                                               logging_test_utils::testLoggingCallback);

    return RUN_ALL_TESTS();
}
