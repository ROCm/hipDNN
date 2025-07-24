/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/component_formatter.hpp>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/logging_callback.hpp>

#define HIPDNN_FRONTEND_TESTS "hipdnn_frontend_tests"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    spdlog::drop_all();
    auto test_logger = spdlog::stdout_color_mt(HIPDNN_FRONTEND_TESTS);
    test_logger->set_formatter(std::make_unique<hipdnn::logging::Component_formatter>());
    spdlog::set_level(spdlog::level::info);

    return RUN_ALL_TESTS();
}