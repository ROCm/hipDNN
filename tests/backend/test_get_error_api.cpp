// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

#include "hipdnn_backend.h"
#include "hipdnn_status.h"

// NOLINTBEGIN(modernize-avoid-c-arrays)
TEST(GetErrorStringTest, AllStatusCodes)
{
    std::vector<std::tuple<hipdnnStatus_t, std::string>> status_pairs
        = {{HIPDNN_STATUS_SUCCESS, "HIPDNN_STATUS_SUCCESS"},
           {HIPDNN_STATUS_NOT_INITIALIZED, "HIPDNN_STATUS_NOT_INITIALIZED"},
           {HIPDNN_STATUS_BAD_PARAM, "HIPDNN_STATUS_BAD_PARAM"},
           {HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "HIPDNN_STATUS_BAD_PARAM_NULL_POINTER"},
           {HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED, "HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED"},
           {HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND, "HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND"},
           {HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT, "HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT"},
           {HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH, "HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH"},
           {HIPDNN_STATUS_NOT_SUPPORTED, "HIPDNN_STATUS_NOT_SUPPORTED"},
           {HIPDNN_STATUS_INTERNAL_ERROR, "HIPDNN_STATUS_INTERNAL_ERROR"},
           {HIPDNN_STATUS_ALLOC_FAILED, "HIPDNN_STATUS_ALLOC_FAILED"},
           {HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
            "HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"},
           {HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
            "HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED"},
           {HIPDNN_STATUS_EXECUTION_FAILED, "HIPDNN_STATUS_EXECUTION_FAILED"},
           {static_cast<hipdnnStatus_t>(-999), "HIPDNN_STATUS_UNKNOWN"}};

    for(const auto& [status, text] : status_pairs)
    {
        const char* str = hipdnnGetErrorString(status);
        ASSERT_STREQ(str, text.c_str());
    }
}

TEST(GetLastErrorStringTest, GetLastError)
{
    char buffer[256];
    hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    hipdnnGetLastErrorString(buffer, sizeof(buffer));
    ASSERT_NE(std::string(buffer), "");
}

TEST(GetLastErrorStringTest, NullBufferOrZeroSize)
{
    // Should not crash or throw
    hipdnnGetLastErrorString(nullptr, 10);
    char buf[8];
    hipdnnGetLastErrorString(buf, 0);
}

TEST(GetLastErrorStringTest, BufferTruncationAndNullTermination)
{
    hipdnnStatus_t status = hipdnnDestroy(nullptr);
    ASSERT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    char buffer[2];
    hipdnnGetLastErrorString(buffer, sizeof(buffer));

    ASSERT_EQ(buffer[sizeof(buffer) - 1], '\0');
    ASSERT_LT(strlen(buffer), sizeof(buffer));
    std::string buffer_str(buffer);
    ASSERT_NE(buffer_str, "");
}

TEST(GetLastErrorStringTest, PerThreadErrorIsolation)
{
    // Set error in main thread
    hipdnnDestroy(nullptr);
    char main_buf[256];
    hipdnnGetLastErrorString(main_buf, sizeof(main_buf));

    std::string thread_error;
    std::thread t([&thread_error]() {
        char buf[256];
        hipdnnGetLastErrorString(buf, sizeof(buf));
        thread_error = buf;
    });
    t.join();

    // Main thread error should be unchanged
    char main_buf2[256];
    hipdnnGetLastErrorString(main_buf2, sizeof(main_buf2));
    ASSERT_STREQ(main_buf, main_buf2);
    ASSERT_TRUE(thread_error.empty());
}

TEST(GetLastErrorStringTest, BufferLargerThanMax)
{
    hipdnnDestroy(nullptr);
    char main_buf[1028];
    hipdnnGetLastErrorString(main_buf, sizeof(main_buf));

    char main_buf2[256];
    hipdnnGetLastErrorString(main_buf2, sizeof(main_buf2));
    ASSERT_STREQ(main_buf, main_buf2);
}
// NOLINTEND(modernize-avoid-c-arrays)
