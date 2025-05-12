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
// NOLINTEND