// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "test_util.hpp"

#include <gtest/gtest.h>

#include <vector>

// TODO - Once PluginManager is implemented, we will need to add a testing plugin for this API.
// NOLINTBEGIN(readability-function-cognitive-complexity)
namespace
{
constexpr hipdnnBackendHeurMode_t FALLBACK_MODE = HIPDNN_HEUR_MODE_FALLBACK;
}

class Engine_heuristic_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine_heuristic = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override
    {
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &_engine_heuristic),
            HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engine_heuristic, nullptr);
    }

    void TearDown() override
    {
        if(_engine_heuristic != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine_heuristic), HIPDNN_STATUS_SUCCESS);
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    void set_heuristic_mode()
    {
        ASSERT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                            HIPDNN_ATTR_ENGINEHEUR_MODE,
                                            HIPDNN_TYPE_HEUR_MODE,
                                            1,
                                            &FALLBACK_MODE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void set_operation_graph()
    {
        test_util::create_test_graph(&_graph, _handle);
        ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                            HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &_graph),
                  HIPDNN_STATUS_SUCCESS);
    }

    void populate_engine_heuristic(bool finalize = true)
    {
        set_operation_graph();
        set_heuristic_mode();

        if(finalize)
        {
            ASSERT_EQ(hipdnnBackendFinalize(_engine_heuristic), HIPDNN_STATUS_SUCCESS);
        }
    }
};

TEST_F(Engine_heuristic_api_tests, SetEngineHeuristicOperationGraph)
{
    hipdnnBackendDescriptor_t null_graph = nullptr;
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &null_graph),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::create_test_graph(&_graph, _handle);
    ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_heuristic_api_tests, SetEngineHeuristicMode)
{
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _engine_heuristic, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        2,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_BAD_PARAM);

    auto unsupported_mode = static_cast<hipdnnBackendHeurMode_t>(999);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &unsupported_mode),
              HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_heuristic_api_tests, SetUnsupportedAttribute)
{
    int32_t dummy = 0;
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine_heuristic, HIPDNN_ATTR_ENGINEHEUR_RESULTS, HIPDNN_TYPE_INT32, 1, &dummy),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_api_tests, FinalizeEngineHeuristic)
{
    EXPECT_EQ(hipdnnBackendFinalize(_engine_heuristic), HIPDNN_STATUS_BAD_PARAM);

    set_operation_graph();
    EXPECT_EQ(hipdnnBackendFinalize(_engine_heuristic), HIPDNN_STATUS_BAD_PARAM);

    set_heuristic_mode();
    EXPECT_EQ(hipdnnBackendFinalize(_engine_heuristic), HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_engine_heuristic), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(Engine_heuristic_api_tests, SetAttributeOnFinalizedDescriptor)
{
    populate_engine_heuristic(true);

    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &FALLBACK_MODE),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_heuristic_api_tests, GetAttributeOnUnfinalizedDescriptor)
{
    hipdnnBackendDescriptor_t dummy_graph = nullptr;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &dummy_graph),
              HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(Engine_heuristic_api_tests, GetOperationGraph)
{
    populate_engine_heuristic(true);

    hipdnnBackendDescriptor_t retrieved_graph = nullptr;
    hipdnnBackendDescriptor_t retrieved_graph2 = nullptr;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &retrieved_graph),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        2,
                                        nullptr,
                                        &retrieved_graph),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &retrieved_graph),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(retrieved_graph, nullptr);

    hipdnnBackendDestroyDescriptor(retrieved_graph);
    retrieved_graph = nullptr;

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        &retrieved_graph2),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    hipdnnBackendDestroyDescriptor(retrieved_graph2);
    retrieved_graph2 = nullptr;
}

TEST_F(Engine_heuristic_api_tests, GetHeuristicMode)
{
    populate_engine_heuristic(true);

    hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        2,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        &mode),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &count,
                                        &mode),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(mode, HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(Engine_heuristic_api_tests, GetUnsupportedAttribute)
{
    populate_engine_heuristic(true);

    hipdnnBackendHeurMode_t dummy;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        nullptr,
                                        &dummy),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_heuristic_api_tests, GetEngineConfigsCountOnly)
{
    populate_engine_heuristic(true);

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_INT64,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
}

TEST_F(Engine_heuristic_api_tests, GetEngineConfigs)
{
    populate_engine_heuristic(true);

    std::vector<hipdnnBackendDescriptor_t> configs(3);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        3,
                                        nullptr,
                                        configs.data()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &count,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    for(size_t i = 0; std::cmp_less(i, count); ++i)
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]),
                  HIPDNN_STATUS_SUCCESS);
    }

    count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        3, // Ask for 3, but only one engine avaliable.
                                        &count,
                                        configs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1); // Only one returned since there isn't more than that.

    EXPECT_EQ(hipdnnBackendFinalize(configs[0]), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t engine = nullptr;
    EXPECT_EQ(hipdnnBackendGetAttribute(configs[0],
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &engine),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(engine, nullptr);

    int64_t engine_id = 0;
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(engine_id, -1);

    // Expecting to only need to clean-up 1 engine config, since we only created & requested 1.
    for(auto config : configs)
    {
        if(config != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(config), HIPDNN_STATUS_SUCCESS);
        }
    }

    EXPECT_EQ(hipdnnBackendDestroyDescriptor(engine), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_heuristic_api_tests, GetEngineConfigsRequestMoreThanAvailable)
{
    populate_engine_heuristic(true);

    std::vector<hipdnnBackendDescriptor_t> configs(5);
    for(size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[i]),
                  HIPDNN_STATUS_SUCCESS);
    }

    int64_t count = 0;
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_heuristic,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        5,
                                        &count,
                                        configs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);

    for(auto& config : configs)
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(config), HIPDNN_STATUS_SUCCESS);
    }
}
// NOLINTEND(readability-function-cognitive-complexity)
