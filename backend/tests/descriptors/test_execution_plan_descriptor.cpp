// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "mocks/mock_engine_plugin_resource_manager.hpp"
#include "mocks/mock_handle.hpp"
#include "test_descriptor_utils.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace test_descriptor_utils;
using namespace ::testing;

using ::testing::Return;

class Execution_plan_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<hipdnnBackendDescriptor> _plan_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_config_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_config_bad_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_wrong_type_wrapper = nullptr;
    std::unique_ptr<Mock_handle> _mock_handle = nullptr;
    std::shared_ptr<Mock_engine_plugin_resource_manager> _mock_engine_plugin_resource_manager
        = nullptr;

    std::shared_ptr<Execution_plan_descriptor> get_execution_plan_descriptor() const
    {
        return _plan_wrapper->as_descriptor<Execution_plan_descriptor>();
    }

    std::shared_ptr<Mock_graph_descriptor> get_mock_graph() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_graph_descriptor>(
            _mock_graph_wrapper.get());
    }

    std::shared_ptr<Mock_engine_descriptor> get_mock_engine() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_engine_descriptor>(
            _mock_engine_wrapper.get());
    }

    std::shared_ptr<Mock_engine_config_descriptor> get_mock_engine_config() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_engine_config_descriptor>(
            _mock_engine_config_wrapper.get());
    }

    std::shared_ptr<Mock_engine_config_descriptor> get_mock_engine_config_bad_type() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_engine_config_descriptor>(
            _mock_engine_config_bad_type_wrapper.get());
    }

    static hipdnnEnginePluginExecutionContext_t get_execution_context()
    {
        return reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(0xFFFFFFFF);
    }

    void set_handle()
    {
        EXPECT_CALL(*_mock_engine_plugin_resource_manager, create_execution_context(_, _, _))
            .WillOnce(Return(get_execution_context()));
        EXPECT_CALL(*_mock_engine_plugin_resource_manager, destroy_execution_context(_, _));

        EXPECT_CALL(*_mock_handle, get_plugin_resource_manager())
            .WillOnce(Return(_mock_engine_plugin_resource_manager));
        get_execution_plan_descriptor()->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &_mock_handle);
    }

    void set_engine_config()
    {
        EXPECT_CALL(*get_mock_engine(), get_engine_id()).WillOnce(Return(ENGINE_ID));
        EXPECT_CALL(*get_mock_engine(), get_graph()).WillOnce(Return(get_mock_graph()));

        EXPECT_CALL(*get_mock_engine_config(), is_finalized()).WillOnce(Return(true));
        EXPECT_CALL(*get_mock_engine_config(), get_engine()).WillOnce(Return(get_mock_engine()));
        EXPECT_CALL(*get_mock_engine_config(), get_serialized_engine_config())
            .WillOnce(Invoke([]() { return hipdnnPluginConstData_t{nullptr, 0}; }));

        get_execution_plan_descriptor()->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &_mock_engine_config_wrapper);
    }

    void make_execution_plan_finalized()
    {
        set_handle();
        set_engine_config();
        ASSERT_NO_THROW(get_execution_plan_descriptor()->finalize());
    }

protected:
    void SetUp() override
    {
        _plan_wrapper = create_descriptor<Execution_plan_descriptor>();
        _mock_graph_wrapper = test_descriptor_utils::create_descriptor<Mock_graph_descriptor>();
        _mock_engine_wrapper = create_descriptor<Mock_engine_descriptor>();
        _mock_engine_config_wrapper = create_descriptor<Mock_engine_config_descriptor>();
        _mock_engine_config_bad_type_wrapper = create_descriptor<Mock_engine_config_descriptor>();
        _mock_wrong_type_wrapper = create_descriptor<Mock_engine_descriptor>();
        _mock_handle = std::make_unique<Mock_handle>();
        _mock_engine_plugin_resource_manager
            = std::make_shared<Mock_engine_plugin_resource_manager>();
    }

    void TearDown() override {}

private:
    static constexpr int64_t ENGINE_ID = 0;
};

TEST_F(Execution_plan_descriptor_test, CreateExecutionPlanDescriptor)
{
    auto plan = get_execution_plan_descriptor();
    ASSERT_NE(plan, nullptr);
    ASSERT_FALSE(plan->is_finalized());
    ASSERT_EQ(plan->get_type(), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
}

TEST_F(Execution_plan_descriptor_test, SetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    auto plan = get_execution_plan_descriptor();
    uint64_t dummy_workspace_size;

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size),
        HIPDNN_STATUS_NOT_SUPPORTED);

    make_execution_plan_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &dummy_workspace_size),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorHandle)
{
    auto plan = get_execution_plan_descriptor();
    hipdnnHandle_t handle = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_INT64, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 2, &handle),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
}

TEST_F(Execution_plan_descriptor_test, SetExecutionPlanDescriptorEngineConfig)
{
    auto plan = get_execution_plan_descriptor();
    auto mock_engine_config = get_mock_engine_config();

    EXPECT_CALL(*get_mock_engine_config_bad_type(), is_finalized()).Times(1);
    EXPECT_CALL(*mock_engine_config, is_finalized()).WillOnce(Return(false)).WillOnce(Return(true));

    // is_finalized()->false
    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_engine_config_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    // is_finalized()->true
    plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                        1,
                        &_mock_engine_config_wrapper);

    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_HANDLE,
                                                   1,
                                                   &_mock_engine_config_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   2,
                                                   &_mock_engine_config_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        plan->set_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine_config = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &engine_config),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_engine_config_bad_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(plan->set_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_wrong_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Execution_plan_descriptor_test, FinalizeExecutionPlanDescriptor)
{
    auto plan = get_execution_plan_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(plan->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_handle();
    set_engine_config();

    ASSERT_NO_THROW(plan->finalize());

    ASSERT_THROW(plan->finalize(), hipdnn_backend::Hipdnn_exception);
}

TEST_F(Execution_plan_descriptor_test, GetAttrOnUnfinalizedExecutionPlanDescriptor)
{
    auto plan = get_execution_plan_descriptor();
    uint64_t dummy_workspace_size;

    ASSERT_THROW_HIPDNN_STATUS(plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                   HIPDNN_TYPE_INT64,
                                                   1,
                                                   nullptr,
                                                   &dummy_workspace_size),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorWorkspaceSize)
{
    auto plan = get_execution_plan_descriptor();
    auto mock_engine_config = get_mock_engine_config();
    int64_t workspace_size = 0;

    make_execution_plan_finalized();
    EXPECT_CALL(*mock_engine_config, get_attribute(_, _, _, _, _)).WillOnce(SetArg4ToInt64(1024));
    plan->get_attribute(
        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace_size);
    ASSERT_EQ(workspace_size, 1024);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorEngineConfig)
{
    auto plan = get_execution_plan_descriptor();

    Scoped_descriptor returned_engine_config;
    Scoped_descriptor null_count_engine_config;
    int64_t count = 0;

    make_execution_plan_finalized();

    ASSERT_NO_THROW(plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        returned_engine_config.get_ptr()));

    ASSERT_EQ(count, 1);
    ASSERT_EQ(*returned_engine_config.get(), *(_mock_engine_config_wrapper.get()));

    ASSERT_NO_THROW(plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        null_count_engine_config.get_ptr()));

    ASSERT_EQ(*null_count_engine_config.get(), *(_mock_engine_config_wrapper.get()));
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorEngineConfigErrors)
{
    auto plan = get_execution_plan_descriptor();
    hipdnnBackendDescriptor_t returned_engine_config = nullptr;
    int64_t count = 0;
    void* dummy = &returned_engine_config;

    make_execution_plan_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        plan->get_attribute(
            HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_INT64, 1, &count, &dummy),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   0, // Too small (needs to be at least 1)
                                                   &count,
                                                   &dummy),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Execution_plan_descriptor_test, GetExecutionPlanDescriptorUnsupportedAttr)
{
    auto plan = get_execution_plan_descriptor();
    int64_t count = 0;
    std::array<char, 256> dummy_buffer{};

    make_execution_plan_finalized();

    ASSERT_THROW_HIPDNN_STATUS(plan->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE,
                                                   HIPDNN_TYPE_INT64,
                                                   1,
                                                   &count,
                                                   dummy_buffer.data()),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Execution_plan_descriptor_test, GetEngineConfigThrowsIfNotFinalized)
{
    auto plan = get_execution_plan_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(plan->get_engine_config(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(Execution_plan_descriptor_test, GetEngineConfigReturnsPointerIfFinalized)
{
    auto plan = get_execution_plan_descriptor();
    make_execution_plan_finalized();
    auto engine_config_ptr = plan->get_engine_config();
    ASSERT_NE(engine_config_ptr, nullptr);
    ASSERT_EQ(static_cast<const Backend_descriptor_interface*>(engine_config_ptr.get()),
              static_cast<const Backend_descriptor_interface*>(get_mock_engine_config().get()));
}

TEST_F(Execution_plan_descriptor_test, ExecutionPlanDescriptorGetExecutionContext)
{
    auto plan = get_execution_plan_descriptor();
    make_execution_plan_finalized();
    auto context = plan->get_execution_context();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context, get_execution_context());
}
