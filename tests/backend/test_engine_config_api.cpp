// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "test_util.hpp"

#include <gtest/gtest.h>

class Engine_config_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine_config;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;

    void SetUp() override
    {
        EXPECT_EQ(
            hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engine_config),
            HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engine_config, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine_config), HIPDNN_STATUS_SUCCESS);
        if(_engine != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
        }
    }
};

TEST_F(Engine_config_api_tests, SetEngineConfigEngine)
{
    int64_t gidx = -1; // TODO hardcode for now

    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_config,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    test_util::create_test_engine(&_engine, &_graph, gidx);
    EXPECT_EQ(hipdnnBackendSetAttribute(_engine_config,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &_engine),
              HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_config_api_tests, FinalizeEngineConfig)
{
    int64_t gidx = -1; // TODO hardcode for now

    EXPECT_EQ(hipdnnBackendFinalize(_engine_config), HIPDNN_STATUS_BAD_PARAM);
    test_util::populate_test_engine_config(&_engine_config, &_engine, &_graph, gidx);
    EXPECT_EQ(hipdnnBackendFinalize(_engine_config), HIPDNN_STATUS_SUCCESS);
}

TEST_F(Engine_config_api_tests, GetMaxWorkspaceSizeFromEngineConfig)
{
    int64_t gidx = -1; // TODO hardcode for now
    int64_t max_workspace_size = 0;

    test_util::populate_test_engine_config(&_engine_config, &_engine, &_graph, gidx, true);
    EXPECT_EQ(hipdnnBackendGetAttribute(_engine_config,
                                        HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                        HIPDNN_TYPE_INT64,
                                        1,
                                        nullptr,
                                        &max_workspace_size),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(max_workspace_size, 1024);
}