// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

class hipdnnBackendDescriptor
{
public:
    virtual ~hipdnnBackendDescriptor() = default;
    hipdnnBackendDescriptorType_t type = HIPDNN_INVALID_TYPE;
};

namespace hipdnn_backend
{
class Backend_descriptor : public hipdnnBackendDescriptor
{
public:
    ~Backend_descriptor() override = default;

    virtual hipdnnStatus_t execute(hipdnnHandle_t handle, hipdnnBackendDescriptor_t variant_pack)
        = 0;
    virtual hipdnnStatus_t finalize() = 0;

    virtual hipdnnStatus_t get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t                      requested_element_count,
                                         int64_t*                     element_count,
                                         void*                        array_of_elements)
        = 0;
    virtual hipdnnStatus_t set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t                      element_count,
                                         const void*                  array_of_elements)
        = 0;
};
}