/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/logging_utils.hpp>

#define HIPDNN_SDK_TESTS "hipdnn_sdk_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn::logging::initialize_callback_logging(HIPDNN_SDK_TESTS,
                                                 logging_test_utils::test_logging_callback);

    return RUN_ALL_TESTS();
}
