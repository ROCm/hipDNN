// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "backend_descriptor.hpp"
#include "hipdnn_status_t.h"

namespace hipdnn_backend
{

class Descriptor_factory
{
public:
    static hipdnnStatus_t create(hipdnnBackendDescriptorType_t descriptor_type,
                                 hipdnnBackendDescriptor_t*    descriptor);

    static hipdnnStatus_t create_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                           const uint8_t*             serialized_graph,
                                           size_t                     graph_byte_size);
};

} // namespace hipdnn_backend