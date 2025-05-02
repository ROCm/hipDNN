// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>

class Engine_api_tests : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engine;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &_engine),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_engine, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(Engine_api_tests, CreateEngine) {}