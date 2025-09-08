/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include "logging/Logging.hpp"
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    hipdnn_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_backend::logging::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
