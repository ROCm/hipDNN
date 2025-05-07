// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

class Engine_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Engine_descriptor> _engine = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_graph = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_graph_bad_type = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_graph_unfinished = nullptr;

    void set_graph()
    {
        ASSERT_EQ(_engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &_mock_graph),
                  HIPDNN_STATUS_SUCCESS);
    }

protected:
    void SetUp() override
    {
        _engine = std::make_unique<Engine_descriptor>();

        _mock_graph
            = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, true);

        _mock_graph_bad_type = std::make_unique<Mock_descriptor>();

        _mock_graph_unfinished
            = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    }
};

TEST_F(Engine_descriptor_test, CreateEngineDescriptor)
{
    ASSERT_NE(_engine, nullptr);
}

TEST_F(Engine_descriptor_test, SetEngineDescriptorGraph)
{
    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mock_graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_mock_graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(_engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                      1,
                                                      &_mock_graph_bad_type),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                      1,
                                                      &_mock_graph_unfinished),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_NO_THROW(_engine->set_attribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_graph));
}

TEST_F(Engine_descriptor_test, SetEngineDescriptorGlobalId)
{
    int64_t gidx = 0;

    ASSERT_THROW_HIPDNN_STATUS(_engine->set_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gidx),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 2, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        _engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx));
}
