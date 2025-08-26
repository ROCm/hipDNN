// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle.hpp"

using namespace hipdnn_backend::plugin;

hipdnnHandle::hipdnnHandle()
    : _plugin_resource_manager(EnginePluginResourceManager::create())
{
}

void hipdnnHandle::set_stream(hipStream_t stream)
{
    _stream = stream;
    _plugin_resource_manager->setStream(stream);
}

hipStream_t hipdnnHandle::get_stream() const
{
    return _stream;
}

std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager>
    hipdnnHandle::get_plugin_resource_manager() const
{
    return _plugin_resource_manager;
}
