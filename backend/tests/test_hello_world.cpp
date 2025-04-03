#include <gtest/gtest.h>
#include "hello_world.h"

using namespace hipdnn_backend;

TEST(HelloWorld, getMessage)
{
    ASSERT_STREQ(HelloWorld::getMessage(), "Hello, World!");
}
