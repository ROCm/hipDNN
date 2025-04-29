// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>

TEST(hipDNNHandleAPITests, CreateAndDestroy)
{
    hipdnnHandle_t handle = nullptr;

    hipdnnStatus_t create_status = hipdnnCreate(&handle);
    ASSERT_EQ(create_status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto destroy_status = hipdnnDestroy(handle);
    ASSERT_EQ(destroy_status, HIPDNN_STATUS_SUCCESS);
}

TEST(hipDNNHandleAPITests, SetStreamNullptrHandle)
{
    hipStream_t    test_stream       = nullptr;
    hipdnnStatus_t set_stream_status = hipdnnSetStream(nullptr, test_stream);
    ASSERT_EQ(set_stream_status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(hipDNNHandleAPITests, GetStreamNullptrHandle)
{
    hipStream_t    retrieved_stream  = nullptr;
    hipdnnStatus_t get_stream_status = hipdnnGetStream(nullptr, &retrieved_stream);
    ASSERT_EQ(get_stream_status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(hipDNNHandleAPITests, GetStreamNullptrStreamPointer)
{
    hipdnnHandle_t handle = nullptr;

    hipdnnStatus_t create_status = hipdnnCreate(&handle);
    ASSERT_EQ(create_status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    hipStream_t    test_stream       = nullptr;
    hipdnnStatus_t get_stream_status = hipdnnGetStream(handle, &test_stream);
    ASSERT_EQ(get_stream_status, HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(test_stream, nullptr);

    hipdnnStatus_t destroy_status = hipdnnDestroy(handle);
    ASSERT_EQ(destroy_status, HIPDNN_STATUS_SUCCESS);
}

TEST(hipDNNHandleAPITests, GetStreamPointer)
{
    hipdnnHandle_t handle = nullptr;

    hipdnnStatus_t create_status = hipdnnCreate(&handle);
    ASSERT_EQ(create_status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    hipdnnStatus_t set_stream_status = hipdnnSetStream(handle, stream);
    ASSERT_EQ(set_stream_status, HIPDNN_STATUS_SUCCESS);

    hipStream_t    retrieved_stream  = nullptr;
    hipdnnStatus_t get_stream_status = hipdnnGetStream(handle, &retrieved_stream);
    ASSERT_EQ(get_stream_status, HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(retrieved_stream, stream);

    hipdnnStatus_t destroy_status = hipdnnDestroy(handle);
    ASSERT_EQ(destroy_status, HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}