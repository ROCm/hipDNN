// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_engine_plugin_handle.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateAlsoCreatesMIOpenHandleOnSuccess)
{
    hipdnnEnginePluginHandle_t handle = nullptr;

    auto status = hipdnnEnginePluginCreate(&handle);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopen_handle, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateNonNullHandlePointer)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto status = hipdnnEnginePluginCreate(&handle);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginDestroyNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginSetStreamNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginSetStream(nullptr, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginSetStreamNullStream)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginSetStreamValidStream)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto stream = reinterpret_cast<hipStream_t>(0x1234); // Simulated valid stream
    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, stream), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(handle->stream, stream);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetApplicableEngineIdsNull)
{
    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(nullptr, nullptr, nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetEngineDetailsNull)
{
    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(nullptr, 0, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginDestroyEngineDetailsNull)
{
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetWorkspaceSizeNull)
{
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateExecutionContextNull)
{
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(nullptr, nullptr, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginDestroyExecutionContextNull)
{
    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginExecuteOpGraphNull)
{
    EXPECT_EQ(hipdnnEnginePluginExecuteOpGraph(nullptr, nullptr, nullptr, nullptr, 0),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}
