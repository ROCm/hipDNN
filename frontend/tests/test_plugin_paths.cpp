// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend/backend/backend_wrapper.hpp>
#include <hipdnn_status.h>

using namespace hipdnn_frontend;

TEST(HipdnnBackendWrapperTest, SetPluginPaths_Success)
{
    std::array<const char*, 1> paths = {"./some/valid/path"};

    hipdnnStatus_t status = hipdnn_frontend::hipdnn_backend().set_plugin_paths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE_UNIQUE);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipdnnBackendWrapperTest, SetPluginPaths_BadParams)
{
    hipdnnStatus_t status = hipdnn_frontend::hipdnn_backend().set_plugin_paths_ext(
        0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE_UNIQUE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}