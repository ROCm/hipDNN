// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hello_world.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(HelloWorld, getMessage)
{
    ASSERT_STREQ(Hello_world::get_message(), "Hello, World!");
}
