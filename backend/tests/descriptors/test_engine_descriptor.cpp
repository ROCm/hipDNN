// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/engine_descriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>

#include <memory>

using namespace hipdnn_backend;

class Engine_descriptor_test : public ::testing::Test
{
public:
    std::unique_ptr<Engine_descriptor> _engine = nullptr;

protected:
    void SetUp() override
    {
        _engine = std::make_unique<Engine_descriptor>();
    }
};

TEST_F(Engine_descriptor_test, CreateEngineDescriptor)
{
    ASSERT_NE(_engine, nullptr);
}