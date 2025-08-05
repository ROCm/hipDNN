// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include <iostream>

namespace hipdnn_plugin
{

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
