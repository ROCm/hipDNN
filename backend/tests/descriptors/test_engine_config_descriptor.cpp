// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_config_descriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

class Engine_config_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Engine_config_descriptor> _engine_config = nullptr;

protected:
    void SetUp() override
    {
        _engine_config = std::make_unique<Engine_config_descriptor>();
    }
};

TEST_F(Engine_config_descriptor_test, CreateEngineConfigDescriptor)
{
    ASSERT_NE(_engine_config, nullptr);
}