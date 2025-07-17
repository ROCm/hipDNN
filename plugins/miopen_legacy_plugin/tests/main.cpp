/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_backend.h>
#include <hipdnn_sdk/logging/logger.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn::logging::initialize_callback_logging("miopen_legacy_plugin", hipdnnLoggingCallback_ext);

    return RUN_ALL_TESTS();
}