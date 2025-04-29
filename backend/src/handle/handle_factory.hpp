// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"
#include "hipdnn_status.h"

namespace hipdnn_backend
{

class Handle_factory
{
public:
    static hipdnnStatus_t create_handle(hipdnnHandle_t* handle);
};

} // namespace hipdnn_backend
