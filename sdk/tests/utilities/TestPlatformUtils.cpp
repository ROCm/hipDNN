// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <iostream>

// TODO: Additional tests
TEST(TestPlatformUtils, GetBuildDir)
{
    std::string buildDir = hipdnn_sdk::utilities::getBuildDir();

    // The build directory should not be empty when built with CMake
    EXPECT_FALSE(buildDir.empty());
}
