/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_frontend::initializeFrontendLogging();

#undef COMPONENT_NAME
#define COMPONENT_NAME "public_hipdnn_frontend_tests"

    hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME, hipdnnLoggingCallback_ext);

    return RUN_ALL_TESTS();
}
