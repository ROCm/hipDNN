// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_exception.hpp>

// NOLINTBEGIN
#define ASSERT_THROW_HIPDNN_PLUGIN_STATUS(x, status)                     \
    do                                                                   \
    {                                                                    \
        try                                                              \
        {                                                                \
            (x);                                                         \
            FAIL() << "Expected exception not thrown";                   \
        }                                                                \
        catch(const hipdnn_plugin::Hipdnn_plugin_exception& e)           \
        {                                                                \
            ASSERT_EQ(e.get_status(), status);                           \
        }                                                                \
        catch(...)                                                       \
        {                                                                \
            FAIL() << "Expected hipdnn_plugin::Hipdnn_plugin_exception"; \
        }                                                                \
    } while(0)
// NOLINTEND
