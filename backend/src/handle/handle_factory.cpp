// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle/handle_factory.hpp"
#include "handle/handle.hpp"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

void Handle_factory::create_handle(hipdnnHandle_t* handle)
{
    if(handle == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM, "handle is null.");
    }

    *handle = new hipdnnHandle();
}

} // namespace hipdnn_backend
