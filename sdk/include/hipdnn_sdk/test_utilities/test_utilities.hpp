// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// NOLINTBEGIN(readability-function-cognitive-complexity)
#define SKIP_IF_NO_DEVICES()                                      \
    do                                                            \
    {                                                             \
        int device_count;                                         \
        auto result = hipGetDeviceCount(&device_count);           \
        if(result == hipErrorNoDevice || device_count == 0)       \
        {                                                         \
            GTEST_SKIP() << "No devices available. Skipping test."; \
        }                                                         \
    } while(0)
// NOLINTEND(readability-function-cognitive-complexity)