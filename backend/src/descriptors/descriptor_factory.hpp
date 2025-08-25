// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "backend_descriptor.hpp"
#include "hipdnn_status.h"

namespace hipdnn_backend
{

class DescriptorFactory
{
public:
    static void create(hipdnnBackendDescriptorType_t descriptorType,
                       hipdnnBackendDescriptor_t* descriptor);

    static void createGraphExt(hipdnnBackendDescriptor_t* descriptor,
                               const uint8_t* serializedGraph,
                               size_t graphByteSize);

    static void destroy(hipdnnBackendDescriptor_t descriptor);
};

} // namespace hipdnn_backend
