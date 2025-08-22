// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "types.hpp"
#include <string>

#define HIPDNN_CHECK_ERROR(x) \
    do                        \
    {                         \
        auto err = x;         \
        if(err.is_bad())      \
        {                     \
            return err;       \
        }                     \
    } while(0)

namespace hipdnn_frontend
{
enum class error_code_t
{
    OK,
    INVALID_VALUE,
    HIPDNN_BACKEND_ERROR,
    ATTRIBUTE_NOT_SET
};

typedef struct error_object // NOLINT
{
    error_code_t code;
    std::string err_msg;

    error_object()
        : code(error_code_t::OK)
    {
    }
    error_object(error_code_t error_code, const std::string& message) // NOLINT
        : code(error_code)
        , err_msg(message)
    {
    }

    std::string get_message() const
    {
        return err_msg;
    }

    error_code_t get_code() const
    {
        return code;
    }

    bool is_good() const
    {
        return code == error_code_t::OK;
    }
    bool is_bad() const
    {
        return code != error_code_t::OK;
    }

    bool operator==(error_code_t other_code) const
    {
        return code == other_code;
    }
    bool operator!=(error_code_t other_code) const
    {
        return code != other_code;
    }
    bool operator==(const error_object& other) const
    {
        return code == other.code;
    }
    bool operator!=(const error_object& other) const
    {
        return code != other.code;
    }
} error_t;

#define HIPDNN_RETURN_IF_NE(x, y, error_status, message) \
    if(x != y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_EQ(x, y, error_status, message) \
    if(x == y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_TRUE(x, error_status, message) \
    if(x)                                               \
    {                                                   \
        return {error_status, message};                 \
    }

#define HIPDNN_RETURN_IF_FALSE(x, error_status, message) \
    if(!(x))                                             \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_NULL(x, error_status, message) \
    if(x == nullptr)                                    \
    {                                                   \
        return {error_status, message};                 \
    }

#define HIPDNN_RETURN_IF_LT(x, y, error_status, message) \
    if(x < y)                                            \
    {                                                    \
        return {error_status, message};                  \
    }
}
