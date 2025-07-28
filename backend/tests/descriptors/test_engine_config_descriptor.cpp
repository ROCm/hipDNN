// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/mock_descriptor.hpp"
#include "test_descriptor_utils.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

using ::testing::Return;

class Engine_config_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<hipdnnBackendDescriptor> _engine_config_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_engine_bad_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_wrong_type_wrapper = nullptr;

    std::shared_ptr<Engine_config_descriptor> get_engine_config_descriptor() const
    {
        return _engine_config_wrapper->as_descriptor<Engine_config_descriptor>();
    }

    std::shared_ptr<Mock_descriptor<Engine_descriptor>> get_mock_engine() const
    {
        return _mock_engine_wrapper->as_descriptor<Mock_descriptor<Engine_descriptor>>();
    }

    std::shared_ptr<Mock_descriptor<Engine_descriptor>> get_mock_engine_bad_type() const
    {
        return _mock_engine_bad_type_wrapper->as_descriptor<Mock_descriptor<Engine_descriptor>>();
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

    void set_max_workspace_size() const
    {
        ASSERT_NO_THROW(get_engine_config_descriptor()->set_max_workspace_size(1024));
    }

    void make_engine_config_finalized() const
    {
        set_engine();
        ASSERT_NO_THROW(get_engine_config_descriptor()->finalize());
        set_max_workspace_size();
    }

protected:
    void SetUp() override
    {
        _engine_config_wrapper
            = test_descriptor_utils::create_descriptor<Engine_config_descriptor>();
        _mock_engine_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Engine_descriptor>>();
        _mock_engine_bad_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Engine_descriptor>>();
        _mock_wrong_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Engine_config_descriptor>>();
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

    EXPECT_CALL(*get_mock_engine(), is_finalized()).WillOnce(Return(false));
    ASSERT_THROW_HIPDNN_STATUS(
        engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine_wrapper),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    EXPECT_CALL(*get_mock_engine(), is_finalized()).WillOnce(Return(true));
    ASSERT_NO_THROW(engine_config->set_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine_wrapper));
}

TEST_F(Engine_config_descriptor_test, SetEngineConfigDescriptorMaxWorkspaceSize)
{
    auto engine_config = get_engine_config_descriptor();
    int64_t workspace_size = 1024;
    int64_t bad_workspace_size = -1024;

    ASSERT_THROW_HIPDNN_STATUS(engine_config->set_max_workspace_size(bad_workspace_size),
                               HIPDNN_STATUS_INTERNAL_ERROR);

    ASSERT_THROW_HIPDNN_STATUS(engine_config->set_max_workspace_size(workspace_size),
                               HIPDNN_STATUS_INTERNAL_ERROR);

    make_engine_config_finalized();
    ASSERT_NO_THROW(engine_config->set_max_workspace_size(workspace_size));
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

    set_engine();

    ASSERT_NO_THROW(engine_config->finalize());
    set_max_workspace_size();
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
