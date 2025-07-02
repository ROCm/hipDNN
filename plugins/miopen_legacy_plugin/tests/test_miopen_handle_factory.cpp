// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_engine_plugin_handle.hpp"
#include "miopen_handle_factory.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <miopen/miopen.h>

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

TEST(MiopenHandleFactoryTest, ThrowsOnNullHandle)
{
    EXPECT_THROW(Miopen_handle_factory::create_miopen_handle(nullptr), Hipdnn_plugin_exception);
}

TEST(MiopenHandleFactoryTest, CreatesAndDestroysHandle)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_NO_THROW(Miopen_handle_factory::create_miopen_handle(&handle));
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopen_handle, nullptr);

    // Clean up
    miopenDestroy(handle->miopen_handle);
    delete handle;
}
