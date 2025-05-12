// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

struct hipdnnBackendDescriptor
{
private:
    bool _finalized = false;

public:
    virtual ~hipdnnBackendDescriptor() = default;
    hipdnnBackendDescriptorType_t type = HIPDNN_INVALID_TYPE;
    virtual void finalize()
    {
        _finalized = true;
    }

    virtual bool is_finalized() const
    {
        return _finalized;
    }

    virtual hipdnnStatus_t get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t requested_element_count,
                                         int64_t* element_count,
                                         void* array_of_elements)
        = 0;
    virtual hipdnnStatus_t set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t element_count,
                                         const void* array_of_elements)
        = 0;

    static void pack_descriptor(const hipdnnBackendDescriptor* descriptor, void*& array_of_elements)
    {
        // Utility to convert/pack descriptor into a non-const data pointer.  This is required by the C API,
        // and is done here so the casting can be done in a single place.
        *static_cast<hipdnnBackendDescriptor**>(array_of_elements)
            = const_cast<hipdnnBackendDescriptor*>(descriptor);
    }
};
