// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/execution_plan_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

// NOLINTBEGIN(readability-function-cognitive-complexity)
class Execution_plan_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Execution_plan_descriptor> _plan = nullptr;
    hipdnnHandle_t _handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    std::unique_ptr<Mock_descriptor> _mock_engine_config = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_engine_bad_type = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_engine_unfinished = nullptr;

    void make_execution_plan_finalized()
    {
        ASSERT_NO_THROW(set_handle());
        ASSERT_NO_THROW(set_engine_config());
        ASSERT_NO_THROW(_plan->finalize());
    }

    void set_handle()
    {
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_handle);
    }

    void set_engine_config()
    {
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                             1,
                             &_mock_engine_config);
    }

protected:
    void SetUp() override
    {
        int64_t dummy_workspace_size = 1024;

        _plan = std::make_unique<Execution_plan_descriptor>();

        _mock_engine_config
            = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, true);
        _mock_engine_config->set_data(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size);

        _mock_engine_bad_type = std::make_unique<Mock_descriptor>();

        _mock_engine_unfinished
            = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    }
};

TEST_F(Execution_plan_descriptor_test, CreateExecutionPlanDescriptor)
{
    ASSERT_NE(_plan, nullptr);
    ASSERT_FALSE(_plan->is_finalized());
    ASSERT_EQ(_plan->type, HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
}

TEST_F(Execution_plan_descriptor_test, SetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    uint64_t dummy_workspace_size;

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size),
        HIPDNN_STATUS_NOT_SUPPORTED);

    make_execution_plan_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorHandle)
{
    hipdnnHandle_t handle = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_INT64, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 2, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorEngineConfig)
{
    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_HANDLE, 1, &_mock_engine_config),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    2,
                                                    &_mock_engine_config),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        _plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(_plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mock_engine_bad_type),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    &_mock_engine_unfinished),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    _plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         &_mock_engine_config);
}

TEST_F(Execution_plan_descriptor_test, FinalizeExecutionPlanDescriptor)
{
    ASSERT_THROW_HIPDNN_STATUS(_plan->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_handle();
    set_engine_config();

    ASSERT_NO_THROW(_plan->finalize());

    ASSERT_THROW(_plan->finalize(), hipdnn_backend::Hipdnn_exception);
}

TEST_F(Execution_plan_descriptor_test, GetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    uint64_t dummy_workspace_size;

    ASSERT_THROW_HIPDNN_STATUS(_plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                    HIPDNN_TYPE_INT64,
                                                    1,
                                                    nullptr,
                                                    &dummy_workspace_size),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorWorkspaceSize)
{
    int64_t workspace_size = 0;

    make_execution_plan_finalized();

    _plan->get_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace_size);
    ASSERT_EQ(workspace_size, 1024);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorUnsupportedAttr)
{
    void* dummy;

    make_execution_plan_finalized();

    ASSERT_THROW_HIPDNN_STATUS(_plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                    1,
                                                    nullptr,
                                                    &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}
// NOLINTEND(readability-function-cognitive-complexity)
