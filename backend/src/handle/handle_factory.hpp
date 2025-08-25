// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"
#include "hipdnn_status.h"

namespace hipdnn_backend
{

class HandleFactory
{
public:
    static void createHandle(hipdnnHandle_t* handle);
    static void destroyHandle(hipdnnHandle_t handle);
};

} // namespace hipdnn_backend
