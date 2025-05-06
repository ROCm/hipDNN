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