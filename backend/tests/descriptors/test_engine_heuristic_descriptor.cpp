// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/engine_heuristic_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/scoped_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "mocks/mock_engine_plugin_resource_manager.hpp"
#include "mocks/mock_handle.hpp"
#include "test_descriptor_utils.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace plugin;
using namespace test_descriptor_utils;
using namespace ::testing;

using ::testing::Return;

class Engine_heuristic_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<hipdnnBackendDescriptor> _engine_heuristic_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_graph_bad_type_wrapper = nullptr;
    std::unique_ptr<hipdnnBackendDescriptor> _mock_wrong_type_wrapper = nullptr;
    std::unique_ptr<Mock_handle> _mock_handle = nullptr;
    std::shared_ptr<Mock_engine_plugin_resource_manager> _mock_engine_plugin_resource_manager
        = nullptr;

    std::shared_ptr<Engine_heuristic_descriptor> get_engine_heuristic_descriptor() const
    {
        return _engine_heuristic_wrapper->as_descriptor<Engine_heuristic_descriptor>();
    }

    std::shared_ptr<Mock_graph_descriptor> get_mock_graph() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_graph_descriptor>(
            _mock_graph_wrapper.get());
    }

    std::shared_ptr<Mock_graph_descriptor> get_mock_graph_bad_type() const
    {
        return Mock_descriptor_utility::as_descriptor_unsafe<Mock_graph_descriptor>(
            _mock_graph_bad_type_wrapper.get());
    }

    std::shared_ptr<Mock_descriptor<Engine_heuristic_descriptor>> get_mock_wrong_type() const
    {
        return _mock_wrong_type_wrapper
            ->as_descriptor<Mock_descriptor<Engine_heuristic_descriptor>>();
    }

    void set_graph() const
    {
        EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));
        EXPECT_CALL(*get_mock_graph(), get_handle()).WillRepeatedly(Return(_mock_handle.get()));
        EXPECT_CALL(*_mock_handle, get_plugin_resource_manager())
            .WillRepeatedly(Return(_mock_engine_plugin_resource_manager));
        ASSERT_NO_THROW(
            get_engine_heuristic_descriptor()->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             &_mock_graph_wrapper));
    }

    void set_heuristic_mode() const
    {
        hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        ASSERT_NO_THROW(get_engine_heuristic_descriptor()->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
    }

    void make_engine_heuristic_finalized() const
    {
        set_graph();
        set_heuristic_mode();
        EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_applicable_engine_ids(_))
            .WillRepeatedly(Return(std::vector<int64_t>{0, 1, 2}));
        ASSERT_NO_THROW(get_engine_heuristic_descriptor()->finalize());
    }

protected:
    void SetUp() override
    {
        _engine_heuristic_wrapper = create_descriptor<Engine_heuristic_descriptor>();
        _mock_graph_wrapper = create_descriptor<Mock_graph_descriptor>();
        _mock_graph_bad_type_wrapper = create_descriptor<Mock_graph_descriptor>();
        _mock_wrong_type_wrapper
            = create_descriptor<Mock_descriptor<Engine_heuristic_descriptor>>();
        _mock_handle = std::make_unique<Mock_handle>();
        _mock_engine_plugin_resource_manager
            = std::make_shared<Mock_engine_plugin_resource_manager>();
    }

    void TearDown() override
    {
        _engine_detail_buffers.clear();
    }

    hipdnnPluginConstData_t serialize_engine_details(int64_t gidx)
    {
        flatbuffers::FlatBufferBuilder builder;
        hipdnn_sdk::data_objects::EngineDetailsBuilder engine_details_builder(builder);
        engine_details_builder.add_engine_id(gidx);
        builder.Finish(engine_details_builder.Finish());
        auto engine_details_buffer = builder.Release();
        hipdnnPluginConstData_t serialized_engine_details
            = {.ptr = engine_details_buffer.data(), .size = engine_details_buffer.size()};
        _engine_detail_buffers.push_back(std::move(engine_details_buffer));
        return serialized_engine_details;
    }

    std::vector<flatbuffers::DetachedBuffer> _engine_detail_buffers;
};

TEST_F(Engine_heuristic_descriptor_test, CreateEngineHeuristicDescriptor)
{
    auto heur = get_engine_heuristic_descriptor();
    ASSERT_NE(heur, nullptr);
    ASSERT_FALSE(heur->is_finalized());
    ASSERT_EQ(heur->get_type(), HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorGraph)
{
    auto heur = get_engine_heuristic_descriptor();

    EXPECT_CALL(*get_mock_graph_bad_type(), is_finalized()).Times(1);
    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillOnce(Return(false)).WillOnce(Return(true));

    // is_finalized->false
    ASSERT_THROW_HIPDNN_STATUS(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_graph_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    // is_finalized()->true
    ASSERT_NO_THROW(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_mock_graph_wrapper));

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mock_graph_wrapper),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   2,
                                                   &_mock_graph_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_graph_bad_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    ASSERT_THROW_HIPDNN_STATUS(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_wrong_type_wrapper),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorHeurMode)
{
    auto heur = get_engine_heuristic_descriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    auto unsupported_mode = static_cast<hipdnnBackendHeurMode_t>(999);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &unsupported_mode),
        HIPDNN_STATUS_NOT_SUPPORTED);

    ASSERT_NO_THROW(
        heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = get_engine_heuristic_descriptor();
    int32_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_descriptor_test, SetAttrOnFinalizedEngineHeuristicDescriptor)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(heur->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &_mock_graph_wrapper),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_heuristic_descriptor_test, FinalizeEngineHeuristicDescriptor)
{
    auto heur = get_engine_heuristic_descriptor();
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_applicable_engine_ids)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_graph();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_heuristic_mode();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_descriptor_test, FinalizeEngineHeuristicDescriptorReverseOrder)
{
    auto heur = get_engine_heuristic_descriptor();
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_applicable_engine_ids)
        .Times(AnyNumber()); //Uninteresting call

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_heuristic_mode();
    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_graph();
    ASSERT_NO_THROW(heur->finalize());

    ASSERT_THROW_HIPDNN_STATUS(heur->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_descriptor_test, GetAttrOnUnfinalizedEngineHeuristicDescriptor)
{
    auto heur = get_engine_heuristic_descriptor();
    hipdnnBackendDescriptor_t dummy_graph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   nullptr,
                                                   &dummy_graph),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorUnsupportedAttr)
{
    auto heur = get_engine_heuristic_descriptor();
    hipdnnBackendHeurMode_t dummy;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorGraph)
{
    auto heur = get_engine_heuristic_descriptor();
    Scoped_descriptor graph;
    Scoped_descriptor graph2;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, graph.get_ptr()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   2,
                                                   nullptr,
                                                   graph.get_ptr()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   nullptr,
                                                   nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        graph.get_ptr()));
    ASSERT_EQ(*graph.get(), *(_mock_graph_wrapper.get()));

    int64_t count;
    ASSERT_NO_THROW(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        graph2.get_ptr()));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorEngineConfigs)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_engine_details(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engine_id, const Graph_descriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serialize_engine_details(engine_id);
            }));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, destroy_engine_details(_, _))
        .WillRepeatedly(Return());

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM);

    int64_t count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = create_descriptor_ptr<Engine_config_descriptor>();
    }

    ASSERT_THROW_HIPDNN_STATUS(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   3,
                                                   nullptr,
                                                   configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(auto config : configs)
    {
        delete config;
    }

    configs.clear();

    Scoped_descriptor single_config(create_descriptor_ptr<Engine_config_descriptor>());

    count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &single_config));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsWithNullConfig)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_engine_details(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engine_id, const Graph_descriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serialize_engine_details(engine_id);
            }));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, destroy_engine_details(_, _))
        .WillRepeatedly(Return());

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    configs[0] = create_descriptor_ptr<Engine_config_descriptor>();
    configs[1] = nullptr;
    configs[2] = create_descriptor_ptr<Engine_config_descriptor>();

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   3,
                                                   &count,
                                                   configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsWithNoEngineIds)
{
    auto heur = get_engine_heuristic_descriptor();
    set_graph();
    set_heuristic_mode();

    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_applicable_engine_ids(_))
        .WillRepeatedly(Return(std::vector<int64_t>{}));

    ASSERT_NO_THROW(heur->finalize());

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = create_descriptor_ptr<Engine_config_descriptor>();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 0);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsRequestMoreThanAvailable)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, get_engine_details(_, _, _))
        .WillRepeatedly(
            Invoke([this](int64_t engine_id, const Graph_descriptor*, hipdnnPluginConstData_t* d) {
                *d = this->serialize_engine_details(engine_id);
            }));
    EXPECT_CALL(*_mock_engine_plugin_resource_manager, destroy_engine_details(_, _))
        .WillRepeatedly(Return());

    std::vector<hipdnnBackendDescriptor_t> configs(5);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = create_descriptor_ptr<Engine_config_descriptor>();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 5, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(auto config : configs)
    {
        delete config;
    }
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsCountOnly)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();

    EXPECT_CALL(*get_mock_graph(), is_finalized()).WillRepeatedly(Return(true));

    int64_t count = 0;
    ASSERT_NO_THROW(heur->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorHeurMode)
{
    auto heur = get_engine_heuristic_descriptor();
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        heur->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(
        heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &mode));
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    ASSERT_NO_THROW(
        heur->get_attribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &count, &mode));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(Engine_heuristic_descriptor_test, GetGraphThrowsIfNotFinalized)
{
    auto heur = get_engine_heuristic_descriptor();
    ASSERT_THROW_HIPDNN_STATUS(heur->get_graph(), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(Engine_heuristic_descriptor_test, GetGraphReturnsPointerIfFinalized)
{
    auto heur = get_engine_heuristic_descriptor();
    make_engine_heuristic_finalized();
    auto graph_ptr = heur->get_graph();
    ASSERT_NE(graph_ptr, nullptr);
    ASSERT_EQ(static_cast<const Backend_descriptor_interface*>(graph_ptr.get()),
              static_cast<const Backend_descriptor_interface*>(get_mock_graph().get()));
}
