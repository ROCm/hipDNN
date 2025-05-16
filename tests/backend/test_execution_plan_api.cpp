// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "test_util.hpp"

#include <gtest/gtest.h>

class Execution_plan_api_tests : public ::testing::Test
{
protected:
    static constexpr int64_t GIDX = -1; // TODO hardcode for now
    hipdnnBackendDescriptor_t _plan;
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine_config = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &_plan),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_plan, nullptr);
    }

    void TearDown() override
    {
        destroy_test_descriptor(_plan);
        destroy_test_handle();
        destroy_test_descriptor(_engine_config);
        destroy_test_descriptor(_engine);
        destroy_test_descriptor(_graph);
    }

private:
    void destroy_test_handle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    static void destroy_test_descriptor(hipdnnBackendDescriptor_t descriptor)
    {
        if(descriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(descriptor), HIPDNN_STATUS_SUCCESS);
            descriptor = nullptr;
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

    test_util::create_test_engine_config(&_engine_config, &_engine, &_graph, GIDX, true);

    EXPECT_EQ(hipdnnBackendSetAttribute(_plan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine_config),
              HIPDNN_STATUS_SUCCESS);
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

    test_util::populate_test_execution_plan(
        &_plan, &_handle, &_engine_config, &_engine, &_graph, GIDX, true);
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _plan, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &size),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(size, 1024);
}

TEST_F(Execution_plan_api_tests, FinalizeExecutionPlan)
{
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  _plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_BAD_PARAM);

    test_util::populate_test_execution_plan(
        &_plan, &_handle, &_engine_config, &_engine, &_graph, GIDX);
    EXPECT_EQ(hipdnnBackendFinalize(_plan), HIPDNN_STATUS_SUCCESS);
}