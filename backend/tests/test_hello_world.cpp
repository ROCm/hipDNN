// tests/test_helloworld.cpp
#include <gtest/gtest.h>
#include "hello_world.h"

TEST(HelloWorld, getMessage) {
    ASSERT_STREQ(HelloWorld::getMessage(), "Hello, World!");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}