// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle/handle_factory.hpp"
#include "handle/handle.hpp"
#include "hipdnn_exception.hpp"
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_backend
{

void Handle_factory::create_handle(hipdnnHandle_t* handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    *handle = new hipdnnHandle();

    HIPDNN_LOG_INFO("Created handle: {:p}", static_cast<void*>(*handle));
}

void Handle_factory::destroy_handle(hipdnnHandle_t handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    delete handle;

    HIPDNN_LOG_INFO("Destroyed handle: {:p}", static_cast<void*>(handle));
}

} // namespace hipdnn_backend
