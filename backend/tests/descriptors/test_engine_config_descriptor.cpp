// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/mock_descriptor.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

class Engine_config_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Engine_config_descriptor> _engine_config = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_engine = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_engine_bad_type = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_engine_unfinished = nullptr;

    void set_engine() const
    {
        ASSERT_NO_THROW(_engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine));
    }

    void set_max_workspace_size() const
    {
        ASSERT_NO_THROW(_engine_config->set_max_workspace_size(1024));
    }

    void make_engine_config_finalized() const
    {
        set_engine();
        ASSERT_NO_THROW(_engine_config->finalize());
        set_max_workspace_size();
    }

protected:
    void SetUp() override
    {
        _engine_config = std::make_unique<Engine_config_descriptor>();

        _mock_engine = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, true);

        _mock_engine_bad_type = std::make_unique<Mock_descriptor>();

        _mock_engine_unfinished
            = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    }
};

TEST_F(Engine_config_descriptor_test, CreateEngineConfigDescriptor)
{
    ASSERT_NE(_engine_config, nullptr);
    ASSERT_FALSE(_engine_config->is_finalized());
    ASSERT_EQ(_engine_config->type, HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
}

TEST_F(Engine_config_descriptor_test, SetEngineConfigDescriptorEngine)
{
    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, &_mock_engine),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mock_engine),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             &_mock_engine_bad_type),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             &_mock_engine_unfinished),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(_engine_config->set_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine));
}

TEST_F(Engine_config_descriptor_test, SetEngineConfigDescriptorMaxWorkspaceSize)
{
    int64_t workspace_size = 1024;
    int64_t bad_workspace_size = -1024;

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->set_max_workspace_size(bad_workspace_size),
                               HIPDNN_STATUS_INTERNAL_ERROR);

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->set_max_workspace_size(workspace_size),
                               HIPDNN_STATUS_INTERNAL_ERROR);

    make_engine_config_finalized();
    ASSERT_NO_THROW(_engine_config->set_max_workspace_size(workspace_size));
}

TEST_F(Engine_config_descriptor_test, SetAttrOnFinalizedEngineConfigDescriptor)
{
    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->set_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_engine),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_config_descriptor_test, FinalizeEngineConfigDescriptor)
{
    ASSERT_THROW_HIPDNN_STATUS(_engine_config->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_engine();

    ASSERT_NO_THROW(_engine_config->finalize());
    set_max_workspace_size();
}

TEST_F(Engine_config_descriptor_test, GetAttrOnUnfinalizedEngineConfigDescriptor)
{
    hipdnnBackendDescriptor_t dummy_engine = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             &dummy_engine),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_config_descriptor_test, GetEngineConfigDescriptorUnsupportedAttr)
{
    hipdnnBackendDescriptor_t dummy = nullptr;

    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO,
                                      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                      1,
                                      nullptr,
                                      &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_config_descriptor_test, GetEngineConfigDescriptorEngine)
{
    hipdnnBackendDescriptor_t engine = nullptr;

    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_INT64, 1, nullptr, &engine),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, nullptr, &engine),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(_engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine));
    ASSERT_EQ(engine, _mock_engine.get());

    int64_t count;
    ASSERT_NO_THROW(_engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &engine));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_config_descriptor_test, GetEngineDescriptorMaxWorkspaceSize)
{
    int64_t workspace_size = 0;

    make_engine_config_finalized();

    ASSERT_THROW_HIPDNN_STATUS(_engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             &workspace_size),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 2, nullptr, &workspace_size),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_config->get_attribute(
            HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(_engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, &workspace_size));
    ASSERT_EQ(workspace_size, 1024);

    int64_t count;
    ASSERT_NO_THROW(_engine_config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, &count, &workspace_size));
    ASSERT_EQ(count, 1);
}
