// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle.hpp"

using namespace hipdnn_backend::plugin;

hipdnnHandle::hipdnnHandle()
    : _pluginResourceManager(Engine_plugin_resource_manager::create())
{
}

void hipdnnHandle::setStream(hipStream_t stream)
{
    _stream = stream;
    _pluginResourceManager->set_stream(stream);
}

hipStream_t hipdnnHandle::getStream() const
{
    return _stream;
}

std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager>
    hipdnnHandle::getPluginResourceManager() const
{
    return _pluginResourceManager;
}
