// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_exception.hpp"

// NOLINTBEGIN
#define ASSERT_THROW_HIPDNN_STATUS(x, status)                      \
    do                                                             \
    {                                                              \
        try                                                        \
        {                                                          \
            (x);                                                   \
            FAIL() << "Expected exception not thrown";             \
        }                                                          \
        catch(const hipdnn_backend::Hipdnn_exception& e)           \
        {                                                          \
            ASSERT_EQ(e.get_status(), status);                     \
        }                                                          \
        catch(...)                                                 \
        {                                                          \
            FAIL() << "Expected hipdnn_backend::Hipdnn_exception"; \
        }                                                          \
    } while(0)

#define EXPECT_HIPDNN_EXCEPTION(x, status)                                       \
    do                                                                           \
    {                                                                            \
        bool exception_thrown = false;                                           \
        hipdnnStatus_t actual_status = HIPDNN_STATUS_SUCCESS;                    \
        try                                                                      \
        {                                                                        \
            (x);                                                                 \
        }                                                                        \
        catch(const hipdnn_backend::Hipdnn_exception& e)                         \
        {                                                                        \
            exception_thrown = true;                                             \
            actual_status = e.get_status();                                      \
        }                                                                        \
        catch(...)                                                               \
        {                                                                        \
            FAIL() << "Expected hipdnn_backend::Hipdnn_exception";               \
        }                                                                        \
        EXPECT_TRUE(exception_thrown) << "Expected exception not thrown";        \
        if(exception_thrown)                                                     \
        {                                                                        \
            EXPECT_EQ(actual_status, status)                                     \
                << "Expected status " << status << " but got " << actual_status; \
        }                                                                        \
    } while(0)
// NOLINTEND
