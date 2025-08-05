// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <string>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace hipdnn_plugin
{

class Hipdnn_plugin_exception : public std::exception
{
public:
    explicit Hipdnn_plugin_exception(hipdnnPluginStatus_t status, std::string message)
        : _status(status)
        , _message(std::move(message))
    {
    }

    const char* what() const noexcept override
    {
        return _message.c_str();
    }

    std::string get_message() const noexcept
    {
        return _message;
    }

    hipdnnPluginStatus_t get_status() const noexcept
    {
        return _status;
    }

private:
    hipdnnPluginStatus_t _status;
    std::string _message;
};

#define PLUGIN_THROW_IF_NE(x, y, failure_status, message)       \
    if(x != y)                                                  \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

#define PLUGIN_THROW_IF_EQ(x, y, failure_status, message)       \
    if(x == y)                                                  \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

#define PLUGIN_THROW_IF_TRUE(x, failure_status, message)        \
    if(x)                                                       \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

#define PLUGIN_THROW_IF_FALSE(x, failure_status, message)       \
    if(!(x))                                                    \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

#define PLUGIN_THROW_IF_NULL(x, failure_status, message)        \
    if(x == nullptr)                                            \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

#define PLUGIN_THROW_IF_LT(x, y, failure_status, message)       \
    if(x < y)                                                   \
    {                                                           \
        throw Hipdnn_plugin_exception(failure_status, message); \
    }

}
