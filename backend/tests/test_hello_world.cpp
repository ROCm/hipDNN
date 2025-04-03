#include "hello_world.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(HelloWorld, getMessage)
{
    ASSERT_STREQ(HelloWorld::getMessage(), "Hello, World!");
}
