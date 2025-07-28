// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "test_descriptor_utils.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

using ::testing::Return;

class Engine_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<hipdnnBackendDescriptor> _engine_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_bad_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_wrong_type_wrapper = nullptr;

    Engine_descriptor* get_engine_descriptor() const
    {
        return _engine_wrapper->as_descriptor<Engine_descriptor>().get();
    }

    Mock_descriptor<Graph_descriptor>* get_mock_graph() const
    {
        return _mock_graph_wrapper->as_descriptor<Mock_descriptor<Graph_descriptor>>().get();
    }

    Mock_descriptor<Graph_descriptor>* get_mock_graph_bad_type() const
    {
        return _mock_graph_bad_type_wrapper->as_descriptor<Mock_descriptor<Graph_descriptor>>()
            .get();
    }

    void set_graph() const
    {
        EXPECT_CALL(*get_mock_graph(), is_finalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(get_engine_descriptor()->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &_mock_graph_wrapper));
    }

    void set_global_index() const
    {
        int64_t gidx = 0;
        ASSERT_NO_THROW(get_engine_descriptor()->set_attribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx));
    }

    void make_engine_finalized() const
    {
        set_graph();
        set_global_index();
        ASSERT_NO_THROW(get_engine_descriptor()->finalize());
    }

protected:
    void SetUp() override
    {
        _engine_wrapper = test_descriptor_utils::create_descriptor<Engine_descriptor>();
        _mock_graph_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Graph_descriptor>>();
        _mock_graph_bad_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Graph_descriptor>>();
        _mock_wrong_type_wrapper
            = test_descriptor_utils::create_descriptor<Mock_descriptor<Engine_descriptor>>();
    }
};

TEST_F(Engine_descriptor_test, CreateEngineDescriptor)
{
    auto engine = get_engine_descriptor();
    ASSERT_NE(engine, nullptr);
    ASSERT_FALSE(engine->is_finalized());
    ASSERT_EQ(engine->get_type(), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
}

TEST_F(Engine_descriptor_test, SetEngineDescriptorGraph)
{
    auto engine = get_engine_descriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mock_graph_wrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     2,
                                                     &_mock_graph_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     &_mock_graph_bad_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     &_mock_wrong_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillOnce(Return(false));
    ASSERT_THROW_HIPDNN_STATUS(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     &_mock_graph_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillOnce(Return(true));
    ASSERT_NO_THROW(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &_mock_graph_wrapper));
}

TEST_F(Engine_descriptor_test, SetEngineDescriptorGlobalId)
{
    auto engine = get_engine_descriptor();
    int64_t gidx = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 2, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        engine->set_attribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx));
}

TEST_F(Engine_descriptor_test, SetAttrOnFinalizedEngineDescriptor)
{
    auto engine = get_engine_descriptor();
    make_engine_finalized();

    ASSERT_THROW_HIPDNN_STATUS(engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     &_mock_graph_wrapper),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_descriptor_test, FinalizeEngineDescriptor)
{
    auto engine = get_engine_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_graph();
    set_global_index();

    ASSERT_NO_THROW(engine->finalize());

    ASSERT_THROW_HIPDNN_STATUS(engine->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_descriptor_test, GetAttrOnUnfinalizedEngineDescriptor)
{
    auto engine = get_engine_descriptor();
    hipdnnBackendDescriptor_t dummy_graph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(engine->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     nullptr,
                                                     &dummy_graph),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_descriptor_test, GetEngineDescriptorUnsupportedAttr)
{
    auto engine = get_engine_descriptor();
    int32_t dummy;

    make_engine_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->get_attribute(
            HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET, HIPDNN_TYPE_INT32, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_descriptor_test, GetEngineDescriptorGraph)
{
    auto engine = get_engine_descriptor();
    Scoped_descriptor graph;
    Scoped_descriptor graph_2;

    make_engine_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->get_attribute(
            HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, graph.get_ptr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     2,
                                                     nullptr,
                                                     graph.get_ptr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(engine->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     nullptr,
                                                     nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          nullptr,
                                          graph.get_ptr()));
    ASSERT_EQ(*graph.get(), *(_mock_graph_wrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(engine->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &count,
                                          graph_2.get_ptr()));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_descriptor_test, GetEngineDescriptorGlobalId)
{
    auto engine = get_engine_descriptor();
    int64_t gidx = -1;

    make_engine_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        engine->get_attribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->get_attribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 2, nullptr, &gidx),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        engine->get_attribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(engine->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &gidx));
    ASSERT_EQ(gidx, 0);

    int64_t count;
    ASSERT_NO_THROW(engine->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &count, &gidx));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_descriptor_test, GetGraphThrowsIfNotFinalized)
{
    auto engine = get_engine_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->get_graph(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(Engine_descriptor_test, GetGraphReturnsPointerIfFinalized)
{
    auto engine = get_engine_descriptor();
    make_engine_finalized();
    auto graph_ptr = engine->get_graph();
    ASSERT_NE(graph_ptr, nullptr);
    ASSERT_EQ(static_cast<const Backend_descriptor_interface*>(graph_ptr.get()),
              static_cast<const Backend_descriptor_interface*>(get_mock_graph()));
}

TEST_F(Engine_descriptor_test, GetEngineIdThrowsIfNotFinalized)
{
    auto engine = get_engine_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(engine->get_engine_id(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(Engine_descriptor_test, GetEngineIdReturnsValueIfFinalized)
{
    auto engine = get_engine_descriptor();
    make_engine_finalized();
    auto engine_id = engine->get_engine_id();
    ASSERT_EQ(engine_id, 0);
}
