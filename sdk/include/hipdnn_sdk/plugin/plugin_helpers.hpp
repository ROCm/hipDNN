// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include <iostream>

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__ __VA_OPT__(, ) __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name __VA_OPT__(, ) __VA_ARGS__)

namespace hipdnn_plugin
{

template <typename T>
void throw_if_null(T* value)
{
    if(value == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                      std::string(typeid(T).name()) + " is nullptr");
    }
}

template <class F>
hipdnnPluginStatus_t try_catch(F f)
{
    try
    {
        f();
    }
    catch(const Hipdnn_plugin_exception& ex)
    {
        return Plugin_last_error_manager::set_last_error(ex.get_status(), ex.what());
    }
    catch(const std::exception& ex)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         ex.what());
    }
    catch(...)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         "Unknown exception occured");
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}
} // namespace hipdnn_plugin
