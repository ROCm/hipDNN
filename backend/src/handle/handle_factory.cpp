// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle/handle_factory.hpp"
#include "handle/handle.hpp"

namespace hipdnn_backend
{

hipdnnStatus_t Handle_factory::create_handle(hipdnnHandle_t* handle)
{
    if(handle == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    try
    {
        *handle = new Handle();
        return HIPDNN_STATUS_SUCCESS;
    }
    catch(const std::bad_alloc&)
    {
        return HIPDNN_STATUS_ALLOC_FAILED;
    }
}

} // namespace hipdnn_backend
