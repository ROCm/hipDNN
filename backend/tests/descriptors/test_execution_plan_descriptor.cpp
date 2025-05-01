// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/execution_plan_descriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/mock_descriptor.hpp"

#include <gtest/gtest.h>

using namespace hipdnn_backend;

class Execution_plan_descriptor_test : public ::testing::Test
{
public:
    Execution_plan_descriptor* _plan                 = nullptr;
    hipdnnHandle_t             _handle               = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    Mock_descriptor*           _mock_engine_config   = nullptr;
    Mock_descriptor*           _mock_engine_bad_type = nullptr;
    Mock_descriptor*           _mock_engine_unfinished = nullptr;

    void make_execution_plan_finalized()
    {
        auto status = set_handle();
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

        status = set_engine_config();
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

        status = _plan->finalize();
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    }

    hipdnnStatus_t set_handle()
    {
        return _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle);
    }

    hipdnnStatus_t set_engine_config()
    {
        return _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    &_mock_engine_config);
    }

protected:
    void SetUp() override
    {
        int64_t dummy_workspace_size = 1024;

        _plan = new Execution_plan_descriptor();

        _mock_engine_config = new Mock_descriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, true);
        auto status         = _mock_engine_config->set_data(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

        _mock_engine_bad_type = new Mock_descriptor();

        _mock_engine_unfinished = new Mock_descriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    }
    void TearDown() override
    {
        delete _plan;
        delete _mock_engine_config;
        delete _mock_engine_bad_type;
        delete _mock_engine_unfinished;
    }
};

TEST_F(Execution_plan_descriptor_test, CreateExecutionPlanDescriptor)
{
    ASSERT_NE(_plan, nullptr);
}

TEST_F(Execution_plan_descriptor_test, SetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    uint64_t dummy_workspace_size;

    auto status = _plan->set_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);

    make_execution_plan_finalized();

    status = _plan->set_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_INITIALIZED); // Plan already finalized
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorHandle)
{
    hipdnnHandle_t handle = nullptr;

    auto status
        = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_INT64, 1, &handle);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status
        = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 2, &handle);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status
        = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status
        = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    status
        = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorEngineConfig)
{
    auto status = _plan->set_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_HANDLE, 1, &_mock_engine_config);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  2,
                                  &_mock_engine_config);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = _plan->set_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    status                           = _plan->set_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &_mock_engine_bad_type);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &_mock_engine_unfinished);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    status = _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &_mock_engine_config);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST_F(Execution_plan_descriptor_test, FinalizeExecutionPlanDescriptor)
{
    auto status = _plan->finalize();
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = set_handle();
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    status = set_engine_config();
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = _plan->finalize();
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = _plan->finalize();
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Execution_plan_descriptor_test, GetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    uint64_t dummy_workspace_size;

    auto status = _plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       nullptr,
                                       &dummy_workspace_size);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorWorkspaceSize)
{
    int64_t workspace_size = 0;

    make_execution_plan_finalized();

    auto status = _plan->get_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace_size);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(workspace_size, 1024);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorUnsupportedAttr)
{
    void* dummy;

    make_execution_plan_finalized();

    auto status = _plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       &dummy);
    ASSERT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}