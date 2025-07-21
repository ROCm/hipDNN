// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include <memory>

//NOLINTBEGIN(readability-identifier-naming)
struct hipdnnPrivateBackendDescriptor;

struct hipdnnBackendDescriptor
{
    std::shared_ptr<hipdnnPrivateBackendDescriptor> private_descriptor;
};

struct hipdnnPrivateBackendDescriptor
{
private:
    bool _finalized = false;

public:
    virtual ~hipdnnPrivateBackendDescriptor() = default;
    hipdnnBackendDescriptorType_t type = HIPDNN_INVALID_TYPE;
    virtual void finalize()
    {
        _finalized = true;
    }

    virtual bool is_finalized() const
    {
        return _finalized;
    }

    virtual void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                               hipdnnBackendAttributeType_t attribute_type,
                               int64_t requested_element_count,
                               int64_t* element_count,
                               void* array_of_elements) const
        = 0;
    virtual void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                               hipdnnBackendAttributeType_t attribute_type,
                               int64_t element_count,
                               const void* array_of_elements)
        = 0;

    template <typename Child_descriptor>
    static std::shared_ptr<Child_descriptor> unpack_descriptor(const hipdnnBackendDescriptor* wrapper)
    {
        return std::static_pointer_cast<Child_descriptor>(wrapper->private_descriptor);
    }

    template <typename Child_descriptor>
    static std::shared_ptr<Child_descriptor> unpack_descriptor(const void* array_of_elements)
    {
        static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                      "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");
        
        const auto wrapper = *static_cast<hipdnnBackendDescriptor* const*>(array_of_elements);
        return unpack_descriptor<Child_descriptor>(wrapper);
    }

    template <typename Child_descriptor>
    static void pack_descriptor(const std::shared_ptr<const Child_descriptor>& private_descriptor, void*& array_of_elements)
    {
        static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                      "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

        // Since we return a wrapper to the users, we need to cast the private_descriptor to a shared_ptr of hipdnnPrivateBackendDescriptor
        // and assign it to the backend descriptor's private_descriptor.
        // Due to the C API semantics, we need to const cast in order to return this to the user.
        // Since the backend descriptor is finalized it's expected to be read-only at this point, and still considered const.
        auto descriptor = new hipdnnBackendDescriptor();
        descriptor->private_descriptor = std::const_pointer_cast<hipdnnPrivateBackendDescriptor>(
            std::static_pointer_cast<const hipdnnPrivateBackendDescriptor>(private_descriptor));
        array_of_elements = static_cast<void*>(descriptor);
    }
};
//NOLINTEND(readability-identifier-naming)