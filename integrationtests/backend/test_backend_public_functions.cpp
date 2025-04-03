#include "hipdnn_backend.h"
#include <gtest/gtest.h>

// Test case to verify the functionality of publicFunctionHello
TEST(HipDNNBackendTest, PublicFunctionHelloTest) {
    // Call the function and check its return value
    int result = publicFunctionHello();
    
    // Assuming the function should return 0 for success
    EXPECT_EQ(result, 1337) << "publicFunctionHello did not return the expected value.";
}