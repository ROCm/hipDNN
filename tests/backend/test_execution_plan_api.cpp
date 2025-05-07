// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>

class Execution_plan_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _plan;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine_config = nullptr;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_plan, nullptr);
    }

    void TearDown() override
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

//=====================================
TEST_F(Execution_plan_api_tests, ExecuteWithModifiedVariantPack)
{
    hipdnnBackendDescriptor_t variant_pack = nullptr;

    // Create a handle
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // Create variant pack
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack),
              HIPDNN_STATUS_SUCCESS);

    // Set required attributes on the plan
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_SUCCESS);

    // Execute plan with variant pack
    hipdnnStatus_t status = hipdnnBackendExecute(_handle, _plan, variant_pack);

    EXPECT_TRUE(status == HIPDNN_STATUS_SUCCESS || status == HIPDNN_STATUS_EXECUTION_FAILED
                || status == HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(variant_pack), HIPDNN_STATUS_SUCCESS);

    // The engine config will be destroyed by the framework since we registered it
    // with hipdnnBackendSetAttribute
}