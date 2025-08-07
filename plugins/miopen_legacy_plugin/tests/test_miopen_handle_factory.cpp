// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

#include "hipdnn_engine_plugin_handle.hpp"
#include "miopen_handle_factory.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

TEST(MiopenHandleFactoryTest, ThrowsOnNullHandle)
{
    EXPECT_THROW(Miopen_handle_factory::create_miopen_handle(nullptr), Hipdnn_plugin_exception);
}

TEST(MiopenHandleFactoryTest, CreatesAndDestroysHandle)
{
    SKIP_IF_NO_DEVICES();

    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_NO_THROW(Miopen_handle_factory::create_miopen_handle(&handle));
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopen_handle, nullptr);

    // Clean up
    miopenDestroy(handle->miopen_handle);
    delete handle;
}

TEST(MiopenHandleFactoryTest, ThrowsOnDestroyNullHandle)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_THROW(Miopen_handle_factory::destroy_miopen_handle(handle), Hipdnn_plugin_exception);
}
