// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "backend_descriptor.hpp"
#include "hipdnn_status.h"

namespace hipdnn_backend
{

class Descriptor_factory
{
public:
    static void create(hipdnnBackendDescriptorType_t descriptor_type,
                       hipdnnBackendDescriptor_t* descriptor);

    static void create_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                 const uint8_t* serialized_graph,
                                 size_t graph_byte_size);
};

} // namespace hipdnn_backend