// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

struct hipdnnBackendDescriptor
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

    virtual hipdnnStatus_t execute([[maybe_unused]] hipdnnHandle_t            handle,
                                   [[maybe_unused]] hipdnnBackendDescriptor_t variant_pack)
    {
        // provide default implementation since this is only applicable to the execution plan descriptor
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }

    virtual hipdnnStatus_t finalize() = 0;

    virtual hipdnnStatus_t
        get_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                      [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                      [[maybe_unused]] int64_t                      requested_element_count,
                      [[maybe_unused]] int64_t*                     element_count,
                      [[maybe_unused]] void*                        array_of_elements)
        = 0;
    virtual hipdnnStatus_t
        set_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                      [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                      [[maybe_unused]] int64_t                      element_count,
                      [[maybe_unused]] const void*                  array_of_elements)
        = 0;
};
}