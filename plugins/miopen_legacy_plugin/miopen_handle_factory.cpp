// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

#include "hipdnn_engine_plugin_handle.hpp"
#include "miopen_handle_factory.hpp"

using namespace hipdnn_plugin;

namespace miopen_legacy_plugin
{

void Miopen_handle_factory::create_miopen_handle(hipdnnEnginePluginHandle_t* handle)
{
    if(handle == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    *handle = new hipdnnEnginePluginHandle();

    miopenStatus_t status = miopenCreate(&(*handle)->miopen_handle);

    if(status != miopenStatusSuccess)
    {
        delete *handle;
        *handle = nullptr;
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                      "Failed to create MIOpen handle");
    }
}

void Miopen_handle_factory::destroy_miopen_handle(hipdnnEnginePluginHandle_t handle)
{
    if(handle == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    miopenStatus_t status = miopenDestroy(handle->miopen_handle);

    if(status != miopenStatusSuccess)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                      "Failed to destroy MIOpen handle");
    }
}

} // namespace miopen_legacy_plugin
