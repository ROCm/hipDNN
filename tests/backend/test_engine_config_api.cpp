// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>

class Engine_config_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine_config;

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
    }
};

TEST_F(Engine_config_api_tests, CreateEngine) {}