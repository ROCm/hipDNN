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

TEST_F(Engine_api_tests, CreateEngine) {}

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