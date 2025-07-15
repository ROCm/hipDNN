// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_engine_plugin_handle.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

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

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateTwiceGivesTheSameContainerHandle)
{
    hipdnnEnginePluginHandle_t handle1 = nullptr;
    auto status1 = hipdnnEnginePluginCreate(&handle1);

    hipdnnEnginePluginHandle_t handle2 = nullptr;
    auto status2 = hipdnnEnginePluginCreate(&handle2);

    EXPECT_EQ(status1, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(status2, HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(handle1->miopen_container, handle2->miopen_container);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle1), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle2), HIPDNN_PLUGIN_STATUS_SUCCESS);
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

    auto stream = reinterpret_cast<hipStream_t>(0x1234);
    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, stream), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(handle->stream, stream);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetApplicableEngineIdsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto op_graph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    std::array<int64_t, 1> engine_ids = {0};
    uint32_t num_engines = 0;

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(nullptr, nullptr, nullptr, 0, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  nullptr, op_graph, engine_ids.data(), 1, &num_engines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  handle, nullptr, engine_ids.data(), 1, &num_engines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetApplicableEngineIds(handle, op_graph, nullptr, 1, &num_engines),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(
        hipdnnEnginePluginGetApplicableEngineIds(handle, op_graph, engine_ids.data(), 1, nullptr),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetApplicableEngineIdsValid)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);
    std::array<int64_t, 1> engine_ids = {0};
    uint32_t num_engines = 0;

    //get max 1 engine
    auto status = hipdnnEnginePluginGetApplicableEngineIds(
        handle, &op_graph, engine_ids.data(), 1, &num_engines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(num_engines, 1u);
    EXPECT_EQ(engine_ids[0], 1u);

    engine_ids[0] = 1337;
    status = hipdnnEnginePluginGetApplicableEngineIds(
        handle, &op_graph, engine_ids.data(), 0, &num_engines);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(num_engines, 1u);
    EXPECT_EQ(engine_ids[0], 1337);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
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
