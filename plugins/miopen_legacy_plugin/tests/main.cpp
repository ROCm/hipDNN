/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/logging_utils.hpp>

#define MIOPEN_LEGACY_PLUGIN_TESTS "miopen_legacy_plugin_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    logging_test_utils::initialize_spdlog_default_logger(MIOPEN_LEGACY_PLUGIN_TESTS);

    return RUN_ALL_TESTS();
}
