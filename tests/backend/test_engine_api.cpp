// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "test_util.hpp"

#include <gtest/gtest.h>

class Engine_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine;
    hipdnnBackendDescriptor_t _graph = nullptr;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &_engine),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engine, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
        }
    }
};

TEST_F(Engine_api_tests, SetEngineGraph)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::create_test_graph(&_graph);
    ASSERT_EQ(hipdnnBackendFinalize(_graph), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_graph),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_api_tests, SetEngineGlobalIndex)
{
    int64_t gidx = 0;

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_api_tests, SetEngineAttrNotSupported)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, nullptr),
              HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(Engine_api_tests, SetEngineAttrAlreadyFinalized)
{
    int64_t gidx = 0;

    test_util::populate_test_engine(_engine, &_graph, gidx, true);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(Engine_api_tests, FinalizeEngine)
{
    int64_t gidx = 0;

    EXPECT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_BAD_PARAM);
    test_util::populate_test_engine(_engine, &_graph, gidx);
    EXPECT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_api_tests, GetEngineGraph)
{
    hipdnnBackendDescriptor_t graph;
    int64_t gidx = 0;

    test_util::populate_test_engine(_engine, &_graph, gidx, true);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine,
                                        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        nullptr,
                                        &graph),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(graph, _graph);
}

TEST_F(Engine_api_tests, GetEngineGlobalIndex)
{
    int64_t gidx = 1;
    int64_t gidx_out;

    test_util::populate_test_engine(_engine, &_graph, gidx, true);
    EXPECT_EQ(
        hipdnnBackendGetAttribute(
            _engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &gidx_out),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(gidx_out, gidx);
}