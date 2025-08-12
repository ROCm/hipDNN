// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/mock_descriptor.hpp"
#include "mocks/mock_engine_plugin_resource_manager.hpp"
#include "mocks/mock_handle.hpp"
#include "test_descriptor_utils.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;
using namespace plugin;
using namespace ::testing;

using ::testing::Return;

class Engine_config_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<hipdnnBackendDescriptor> _engine_config_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_bad_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_wrong_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_wrapper = nullptr;
    std::unique_ptr<Mock_handle> _mock_handle = nullptr;
    std::shared_ptr<Mock_engine_plugin_resource_manager> _mock_engine_plugin_resource_manager
        = nullptr;

    std::shared_ptr<Engine_config_descriptor> get_engine_config_descriptor() const
    {
        return _engine_config_wrapper->as_descriptor<Engine_config_descriptor>();
    }

    std::shared_ptr<Mock_engine_descriptor> get_mock_engine() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_engine_descriptor>(
            _mock_engine_wrapper.get());
    }

    std::shared_ptr<Mock_engine_descriptor> get_mock_engine_bad_type() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_engine_descriptor>(
            _mock_engine_bad_type_wrapper.get());
    }

    std::shared_ptr<Mock_graph_descriptor> get_mock_graph_descriptor() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_graph_descriptor>(
            _mock_graph_wrapper.get());
    }

    void set_engine() const
    {
        EXPECT_CALL(*get_mock_engine(), is_finalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(
            get_engine_config_descriptor()->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          1,
                                                          &_mock_engine_wrapper));
    }

    void make_engine_config_finalized() const
    {
        // TODO: These expectations being hidden in here are dangerous.
        // It's easy to forget they exist, and if any of these functions are called prior to this
        // call it is undefined behavior. Similarly, any expectations on functions called within
        // "finalize" or "setup" made after calling this function are UB
        EXPECT_CALL(*get_mock_engine(), is_finalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*get_mock_engine(), get_engine_id()).WillRepeatedly(Return(1));
        EXPECT_CALL(*get_mock_engine(), get_graph())
            .WillRepeatedly(Return(get_mock_graph_descriptor()));
        EXPECT_CALL(*get_mock_graph_descriptor(), get_handle())
            .WillOnce(Return(_mock_handle.get()));
        EXPECT_CALL(*_mock_handle, get_plugin_resource_manager())
            .WillOnce(Return(_mock_engine_plugin_resource_manager));
        EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_workspace_size(_, _, _))
            .WillOnce(Return(1024));

        set_engine();
        ASSERT_NO_THROW(get_engine_config_descriptor()->finalize());
    }

protected:
    void SetUp() override
    {
        _engine_config_wrapper
            = test_descriptor_utils::create_descriptor<Engine_config_descriptor>();
        _mock_engine_wrapper = test_descriptor_utils::create_descriptor<Mock_engine_descriptor>();
        _mock_engine_bad_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_engine_descriptor>();
        _mock_wrong_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Engine_config_descriptor>>();
        _mock_graph_wrapper = test_descriptor_utils::create_descriptor<Mock_graph_descriptor>();
        _mock_handle = std::make_unique<Mock_handle>();
        _mock_engine_plugin_resource_manager
            = std::make_shared<Mock_engine_plugin_resource_manager>();
    }
};

TEST_F(Engine_config_descriptor_test, CreateEngineConfigDescriptor)
{
    auto engine_config = get_engine_config_descriptor();
    ASSERT_NE(engine_config, nullptr);
    ASSERT_FALSE(engine_config->is_finalized());
    ASSERT_EQ(engine_config->get_type(), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(Engine_config_descriptor_test, SetEngineConfigDescriptorEngine)
{
    auto engine_config = get_engine_config_descriptor();

    EXPECT_CALL(*get_mock_engine_bad_type(), is_finalized()).Times(1);
    EXPECT_CALL(*get_mock_engine(), get_engine_id()).Times(AnyNumber());
    EXPECT_CALL(*get_mock_engine(), is_finalized()).WillOnce(Return(false)).WillOnce(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine_wrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(engine_config->set_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine_wrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mock_engine_wrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mock_engine_wrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t engine = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engine_config->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            &_mock_engine_bad_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engine_config->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            &_mock_wrong_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_config_descriptor_test, SetAttrOnFinalizedEngineConfigDescriptor)
{
    auto engine_config = get_engine_config_descriptor();
    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine_wrapper),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_config_descriptor_test, FinalizeEngineConfigDescriptor)
{
    auto engine_config = get_engine_config_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine_config->finalize(), HIPDNN_STATUS_BAD_PARAM);

    make_engine_config_finalized();
}

TEST_F(Engine_config_descriptor_test, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    auto engine_config = get_engine_config_descriptor();
    hipdnnBackendDescriptor_t dummy_engine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            nullptr,
                                                            &dummy_engine),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_config_descriptor_test, GetEngineConfigDescriptorUnsupportedAttr)
{
    auto engine_config = get_engine_config_descriptor();
    hipdnnBackendDescriptor_t dummy = nullptr;

    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            nullptr,
                                                            &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_config_descriptor_test, GetEngineConfigDescriptorEngine)
{
    auto engine_config = get_engine_config_descriptor();
    Scoped_descriptor engine;
    Scoped_descriptor engine_2;
    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, engine.get_ptr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            2,
                                                            nullptr,
                                                            engine.get_ptr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 nullptr,
                                                 engine.get_ptr()));
    ASSERT_EQ(*engine.get(), *(_mock_engine_wrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &count,
                                                 engine_2.get_ptr()));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_config_descriptor_test, GetEngineThrowsIfNotFinalized)
{
    auto engine_config = get_engine_config_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine_config->get_engine(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(Engine_config_descriptor_test, GetEngineReturnsPointerIfFinalized)
{
    auto engine_config = get_engine_config_descriptor();
    make_engine_config_finalized();
    auto engine_ptr = engine_config->get_engine();
    ASSERT_NE(engine_ptr, nullptr);
    ASSERT_EQ(static_cast<const Backend_descriptor_interface*>(engine_ptr.get()),
              static_cast<const Backend_descriptor_interface*>(get_mock_engine().get()));
}

TEST_F(Engine_config_descriptor_test, GetEngineDescriptorMaxWorkspaceSize)
{
    auto engine_config = get_engine_config_descriptor();
    int64_t workspace_size = 0;

    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                            1,
                                                            nullptr,
                                                            &workspace_size),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspace_size),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace_size));
    ASSERT_EQ(workspace_size, 1024);

    int64_t count;
    ASSERT_NO_THROW(engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspace_size));
    ASSERT_EQ(count, 1);
}
