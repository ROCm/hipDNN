/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/logger.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // hipdnn::logging::initialize_logging_based_on_environment_variables("backend_tests");
    // hipdnn::logging::set_log_level("info");

    return RUN_ALL_TESTS();
}