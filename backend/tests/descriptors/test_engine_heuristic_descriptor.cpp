// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/engine_heuristic_descriptor.hpp"
#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include "mocks/mock_descriptor.hpp"
#include "test_macros.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;

using ::testing::Return;

class Engine_heuristic_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Engine_heuristic_descriptor> _engine_heuristic = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_graph = nullptr;
    std::unique_ptr<Mock_descriptor> _mock_graph_bad_type = nullptr;

    void set_graph() const
    {
        EXPECT_CALL(*_mock_graph, is_finalized()).WillOnce(Return(true));
        ASSERT_NO_THROW(_engine_heuristic->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         &_mock_graph));
    }

    void set_heuristic_mode() const
    {
        hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        ASSERT_NO_THROW(_engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
    }

    void set_engine_ids() const
    {
        std::vector<int64_t> engine_ids = {0, 1, 2};
        ASSERT_NO_THROW(_engine_heuristic->set_engine_ids(engine_ids));
    }

    void make_engine_heuristic_finalized() const
    {
        set_graph();
        set_heuristic_mode();
        ASSERT_NO_THROW(_engine_heuristic->finalize());

        // Engine Ids are set after finalization, and aren't set through the API.
        set_engine_ids();
    }

protected:
    void SetUp() override
    {
        _engine_heuristic = std::make_unique<Engine_heuristic_descriptor>();

        _mock_graph = std::make_unique<Mock_descriptor>(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        _mock_graph_bad_type = std::make_unique<Mock_descriptor>();
    }

    static void destroy_config(Engine_config_descriptor* config)
    {
        if(config != nullptr)
        {
            // TODO - fix needing to delete internal engine descriptor to prevent leaks
            config->finalize();
            Engine_descriptor* engine = nullptr;
            config->get_attribute(
                HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);
            delete engine;
            delete config;
        }
    }
};

TEST_F(Engine_heuristic_descriptor_test, CreateEngineHeuristicDescriptor)
{
    ASSERT_NE(_engine_heuristic, nullptr);
    ASSERT_FALSE(_engine_heuristic->is_finalized());
    ASSERT_EQ(_engine_heuristic->type, HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorGraph)
{
    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, &_mock_graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         2,
                                         &_mock_graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &_mock_graph_bad_type),
        HIPDNN_STATUS_BAD_PARAM);

    EXPECT_CALL(*_mock_graph, is_finalized()).WillOnce(Return(false));
    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &_mock_graph),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);

    EXPECT_CALL(*_mock_graph, is_finalized()).WillOnce(Return(true));
    ASSERT_NO_THROW(_engine_heuristic->set_attribute(
        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_mock_graph));
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorHeurMode)
{
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    auto unsupported_mode = static_cast<hipdnnBackendHeurMode_t>(999);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->set_attribute(
                                   HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, &mode),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->set_attribute(
                                   HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &unsupported_mode),
        HIPDNN_STATUS_NOT_SUPPORTED);

    ASSERT_NO_THROW(_engine_heuristic->set_attribute(
        HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode));
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineHeuristicDescriptorUnsupportedAttr)
{
    int32_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->set_attribute(
                                   HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_descriptor_test, SetEngineIds)
{
    std::vector<int64_t> engine_ids = {0, 1, 2};

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->set_engine_ids(engine_ids),
                               HIPDNN_STATUS_INTERNAL_ERROR);

    make_engine_heuristic_finalized();

    ASSERT_NO_THROW(_engine_heuristic->set_engine_ids(engine_ids));
}

TEST_F(Engine_heuristic_descriptor_test, SetAttrOnFinalizedEngineHeuristicDescriptor)
{
    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->set_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &_mock_graph),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_heuristic_descriptor_test, FinalizeEngineHeuristicDescriptor)
{
    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_graph();
    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_heuristic_mode();
    ASSERT_NO_THROW(_engine_heuristic->finalize());

    set_engine_ids();

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_descriptor_test, FinalizeEngineHeuristicDescriptorReverseOrder)
{
    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_heuristic_mode();
    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);

    set_graph();
    ASSERT_NO_THROW(_engine_heuristic->finalize());

    set_engine_ids();

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_descriptor_test, GetAttrOnUnfinalizedEngineHeuristicDescriptor)
{
    hipdnnBackendDescriptor_t dummy_graph = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         nullptr,
                                         &dummy_graph),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorUnsupportedAttr)
{
    hipdnnBackendHeurMode_t dummy;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorGraph)
{
    hipdnnBackendDescriptor_t graph = nullptr;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_INT64, 1, nullptr, &graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         2,
                                         nullptr,
                                         &graph),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         nullptr,
                                         nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(_engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     nullptr,
                                                     &graph));
    ASSERT_EQ(graph, _mock_graph.get());

    int64_t count;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &graph));
    ASSERT_EQ(count, 1);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorEngineConfigs)
{
    make_engine_heuristic_finalized();
    EXPECT_CALL(*_mock_graph, is_finalized()).WillRepeatedly(Return(true));

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM);

    int64_t count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    std::vector<Engine_config_descriptor*> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = new Engine_config_descriptor();
    }

    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                                3,
                                                                nullptr,
                                                                configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(auto* config : configs)
    {
        destroy_config(config);
    }

    auto* single_config = new Engine_config_descriptor();

    count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &single_config));
    ASSERT_EQ(count, 1);

    destroy_config(single_config);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsWithNullConfig)
{
    make_engine_heuristic_finalized();

    std::vector<Engine_config_descriptor*> configs(3);
    configs[0] = new Engine_config_descriptor();
    configs[1] = nullptr;
    configs[2] = new Engine_config_descriptor();

    EXPECT_CALL(*_mock_graph, is_finalized()).WillRepeatedly(Return(true));
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(_engine_heuristic->get_attribute(HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                                3,
                                                                &count,
                                                                configs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // TODO: Since we partially populate due to the nullptr in the array, only the first config has internal state set.
    //       Need to properly clean-up internal state without shenanigans.
    destroy_config(configs[0]);
    delete configs[2];
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsWithNoEngineIds)
{
    set_graph();
    set_heuristic_mode();

    ASSERT_NO_THROW(_engine_heuristic->finalize());

    std::vector<int64_t> engine_ids = {};
    ASSERT_NO_THROW(_engine_heuristic->set_engine_ids(engine_ids));

    EXPECT_CALL(*_mock_graph, is_finalized()).WillRepeatedly(Return(true));

    std::vector<Engine_config_descriptor*> configs(3);
    for(size_t i = 0; i < 3; ++i)
    {
        configs[i] = new Engine_config_descriptor();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 3, &count, configs.data()));
    ASSERT_EQ(count, 0);

    for(auto* config : configs)
    {
        delete config;
    }
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsRequestMoreThanAvailable)
{
    make_engine_heuristic_finalized();

    EXPECT_CALL(*_mock_graph, is_finalized()).WillRepeatedly(Return(true));

    std::vector<Engine_config_descriptor*> configs(5);
    for(size_t i = 0; i < 5; ++i)
    {
        configs[i] = new Engine_config_descriptor();
    }

    int64_t count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 5, &count, configs.data()));
    ASSERT_EQ(count, 3);

    for(size_t i = 0; i < 5; ++i)
    {
        // TODO: Internal memory not being tracked properly, and we need to manually free.
        if(std::cmp_less(i, count))
        {
            destroy_config(configs[i]);
        }
        else
        {
            delete configs[i];
        }
    }
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineConfigsCountOnly)
{
    make_engine_heuristic_finalized();

    EXPECT_CALL(*_mock_graph, is_finalized()).WillRepeatedly(Return(true));

    int64_t count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
    ASSERT_EQ(count, 3);
}

TEST_F(Engine_heuristic_descriptor_test, GetEngineHeuristicDescriptorHeurMode)
{
    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    make_engine_heuristic_finalized();

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 2, nullptr, &mode),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _engine_heuristic->get_attribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr, &mode));
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    ASSERT_NO_THROW(_engine_heuristic->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &count, &mode));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}
