// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "mock_descriptor.hpp"
#include <gtest/gtest.h>

class Execution_plan_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _plan;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine_config = nullptr;

    void SetUp(/* NOLINT(readability-convert-member-functions-to-static */) override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_plan, nullptr);
    }

    void TearDown(/*NOLINT(readability-convert-member-functions-to-static*/) override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_plan), HIPDNN_STATUS_SUCCESS);
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }
};

TEST_F(Execution_plan_api_tests, SetExecutionPlanHandle)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Execution_plan_api_tests, SetExecutionPlanEngineConfig)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_plan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_handle),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // TODO add more tests when engine_config is created
}

TEST_F(Execution_plan_api_tests, SetExecutionPlanAttrNotSupported)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Execution_plan_api_tests, GetExecutionPlanWorkSpaceSize)
{
    size_t size = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &size),
        HIPDNN_STATUS_NOT_INITIALIZED);

    // TODO add more tests when engine_config is created and this can be finalized
}

TEST_F(Execution_plan_api_tests, FinalizeExecutionPlan)
{
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    // TODO add more tests when engine_config is created
}

TEST_F(Execution_plan_api_tests, ExecuteWithModifiedVariantPack)
{
    hipdnnBackendDescriptor_t engine = nullptr;
    if (_handle == nullptr) {
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &engine),
              HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t graph = nullptr;
    int64_t engine_id = -1; // HIPDNN_ENGINE_ID_FAKE; // -1 for the fake plugin

    test_util::create_test_graph(graph);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipdnnBackendFinalize(graph), HIPDNN_STATUS_SUCCESS);
    
    // only populate since engine has been created in test fixture`
    populate_test_engine(engine, graph, engine_id, true);
    create_test_engine_config(_engine_config, engine, graph, engine_id, true);

    int64_t dummy_workspace_size = 1024;
    auto status = _engine_config->set_data(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size);

    // We can verify it was set correctly:
    int64_t workspace_size = 0;
    ASSERT_EQ(hipdnnBackendGetAttribute(engine_config,
                                        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &workspace_size),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(workspace_size, dummy_workspace_size);

    // 5) Create variant pack
    hipdnnBackendDescriptor_t variant_pack = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack),
              HIPDNN_STATUS_SUCCESS);

    // 6) Set values to variant pack if needed
    // For now, the Fake_plugin doesn't require any values

    // 7) Execute plan with variant pack
    hipdnnStatus_t status = hipdnnBackendExecute(_handle, _plan, variant_pack);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Clean up resources
    EXPECT_EQ(hipdnnBackendDestroyDescriptor(variant_pack), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendDestroyDescriptor(graph), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendDestroyDescriptor(engine), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendDestroyDescriptor(engine_config), HIPDNN_STATUS_SUCCESS);
}