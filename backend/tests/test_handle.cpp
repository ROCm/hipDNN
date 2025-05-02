// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "handle/handle.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

namespace hipdnn_backend
{

TEST(HandleTests, DefaultStreamIsNull)
{
    Handle handle;
    EXPECT_EQ(handle.get_stream(), nullptr) << "Default stream should be nullptr.";
}

TEST(GPU_HandleTests, SetAndGetStream)
{
    SKIP_IF_NO_DEVICES();

    Handle handle;

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    handle.set_stream(stream);

    EXPECT_EQ(handle.get_stream(), stream) << "Stream was not set correctly.";

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}

TEST(GPU_HandleTests, SetStreamToNull)
{
    SKIP_IF_NO_DEVICES();

    Handle handle;

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    handle.set_stream(stream);
    EXPECT_EQ(handle.get_stream(), stream) << "Stream was not set correctly.";

    handle.set_stream(nullptr);
    EXPECT_EQ(handle.get_stream(), nullptr) << "Stream should be nullptr after being reset.";

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}

}