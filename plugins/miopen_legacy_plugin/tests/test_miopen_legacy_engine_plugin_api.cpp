// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_details_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"
#include "mocks/mock_hipdnn_engine_plugin_execution_context.hpp"

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateNullHandle)
{
    EXPECT_EQ(hipdnnEnginePluginCreate(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateAlsoCreatesMIOpenHandleOnSuccess)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;

    auto status = hipdnnEnginePluginCreate(&handle);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->miopen_handle, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateTwiceGivesTheSameContainerHandle)
{
    SKIP_IF_NO_DEVICES();
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
    SKIP_IF_NO_DEVICES();
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
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnEnginePluginSetStream(handle, nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginSetStreamValidStream)
{
    SKIP_IF_NO_DEVICES();
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
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
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
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto op_graph = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    hipdnnPluginConstData_t engine_details_out;

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(nullptr, 1, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(nullptr, 1, op_graph, &engine_details_out),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(handle, 1, nullptr, &engine_details_out),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginGetEngineDetails(handle, 1, op_graph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetEngineDetailsValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);
    hipdnnPluginConstData_t engine_details_out;

    auto status = hipdnnEnginePluginGetEngineDetails(handle, 1, &op_graph, &engine_details_out);

    hipdnn_plugin::Engine_details_wrapper engine_details(engine_details_out.ptr,
                                                         engine_details_out.size);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(engine_details.engine_id(), 1);

    // Clean up
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, &engine_details_out),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginDestroyEngineDetailsNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    hipdnnPluginConstData_t engine_details_out;

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(nullptr, &engine_details_out),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    engine_details_out.ptr = nullptr;
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineDetails(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetWorkspaceSizeNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engine_config = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto op_graph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    size_t workspace_size = 123;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(nullptr, engine_config, op_graph, &workspace_size),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engine_config
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, nullptr, op_graph, &workspace_size),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null op_graph
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, engine_config, nullptr, &workspace_size),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null workspace_size
    EXPECT_EQ(hipdnnEnginePluginGetWorkspaceSize(handle, engine_config, op_graph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginGetWorkspaceSizeValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Create a valid flatbuffer graph and engine config
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    auto engine_config_builder = flatbuffer_test_utils::create_valid_engine_config(1);
    auto serialized_engine_config = engine_config_builder.Release();
    hipdnnPluginConstData_t engine_config
        = flatbuffer_test_utils::create_valid_const_data_engine_config(serialized_engine_config);

    size_t workspace_size = 0;
    auto status
        = hipdnnEnginePluginGetWorkspaceSize(handle, &engine_config, &op_graph, &workspace_size);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_size, 0u); // batchnorm workspace size is always 0

    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    auto engine_config = reinterpret_cast<hipdnnPluginConstData_t*>(0x5678);
    auto op_graph = reinterpret_cast<hipdnnPluginConstData_t*>(0x9abc);
    hipdnnEnginePluginExecutionContext_t execution_context;

    // Null handle
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(
                  nullptr, engine_config, op_graph, &execution_context),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null engine_config
    EXPECT_EQ(
        hipdnnEnginePluginCreateExecutionContext(handle, nullptr, op_graph, &execution_context),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null op_graph
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(
                  handle, engine_config, nullptr, &execution_context),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    // Null execution_context
    EXPECT_EQ(hipdnnEnginePluginCreateExecutionContext(handle, engine_config, op_graph, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginCreateExecutionContextValid)
{
    SKIP_IF_NO_DEVICES();
    hipdnnEnginePluginHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    auto engine_config_builder = flatbuffer_test_utils::create_valid_engine_config(1);
    auto serialized_engine_config = engine_config_builder.Release();
    hipdnnPluginConstData_t engine_config
        = flatbuffer_test_utils::create_valid_const_data_engine_config(serialized_engine_config);

    hipdnnEnginePluginExecutionContext_t execution_context = nullptr;
    auto status = hipdnnEnginePluginCreateExecutionContext(
        handle, &engine_config, &op_graph, &execution_context);

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(execution_context, nullptr);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, execution_context),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(MiopenLegacyEnginePluginApiTest, EnginePluginDestroyExecutionContextNull)
{
    auto handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(0x1234);
    Mock_hipdnn_engine_plugin_execution_context execution_context;

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(nullptr, &execution_context),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}
