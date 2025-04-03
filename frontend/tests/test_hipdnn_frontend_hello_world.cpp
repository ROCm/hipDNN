#include <gtest/gtest.h>
#include "hipdnn_frontend_hello_world.hpp"

TEST(HelloWorldFrontendTest, SayHelloReturnsCorrectMessage) {
    HelloWorldFrontend helloWorld;
    EXPECT_EQ(helloWorld.sayHello(), "Hello, Frontend");
}
