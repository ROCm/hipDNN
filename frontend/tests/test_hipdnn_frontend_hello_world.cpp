#include "hipdnn_frontend_hello_world.hpp"
#include <gtest/gtest.h>

TEST(HelloWorldFrontendTest, SayHelloReturnsCorrectMessage)
{
    HelloWorldFrontend helloWorld;
    EXPECT_EQ(helloWorld.sayHello(), "Hello, Frontend");
}
