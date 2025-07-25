// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include <memory>

//NOLINTBEGIN(readability-identifier-naming)

namespace hipdnn_backend
{
template <typename Child_descriptor>
std::shared_ptr<Child_descriptor> unpack_descriptor(const hipdnnBackendDescriptor* descriptor,
                                                    hipdnnStatus_t error_status,
                                                    const std::string& error_message);
}

struct Backend_descriptor_interface
{
    virtual ~Backend_descriptor_interface() = default;

    virtual void finalize() = 0;
    virtual bool is_finalized() const = 0;
    virtual void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                               hipdnnBackendAttributeType_t attribute_type,
                               int64_t requested_element_count,
                               int64_t* element_count,
                               void* array_of_elements) const = 0;
    virtual void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                               hipdnnBackendAttributeType_t attribute_type,
                               int64_t element_count,
                               const void* array_of_elements) = 0;

    virtual hipdnnBackendDescriptorType_t get_type() const = 0;
};

struct hipdnnBackendDescriptor : public Backend_descriptor_interface
{
    ~hipdnnBackendDescriptor() override = default;

    void finalize() override;
    bool is_finalized() const override;
    void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t requested_element_count,
                       int64_t* element_count,
                       void* array_of_elements) const override;
    void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements) override;

    std::shared_ptr<Backend_descriptor_interface> impl;

    template <typename Child_descriptor>
    std::shared_ptr<Child_descriptor> as_descriptor() const
    {
        return hipdnn_backend::unpack_descriptor<Child_descriptor>(
            this,
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Failed to cast backend descriptor: Null descriptor provided.");
    }

    hipdnnBackendDescriptorType_t get_type() const override;
};

namespace hipdnn_backend
{
template <typename T>
class hipdnnBackendDescriptorImpl : public Backend_descriptor_interface
{
private:
    bool _finalized = false;
    hipdnnBackendDescriptorType_t _type = HIPDNN_INVALID_TYPE;

public:
    hipdnnBackendDescriptorImpl()
    : _type(get_static_type())
    {
    }

    void finalize() override { _finalized = true; }
    bool is_finalized() const override { return _finalized; }

    void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t requested_element_count,
                       int64_t* element_count,
                       void* array_of_elements) const override = 0;
    void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements) override = 0;

    hipdnnBackendDescriptorType_t get_type() const override { return _type; }

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return T::get_static_type();
    }
};

// Helper trait for static_assert
template <typename Child_descriptor>
constexpr bool is_valid_backend_child_descriptor_v =
    std::is_base_of_v<
        hipdnnBackendDescriptorImpl<std::remove_const_t<Child_descriptor>>,
        std::remove_const_t<Child_descriptor>
    >;

// Unpacks a hipdnnBackendDescriptor into a shared_ptr of the specified type.
// Throws an exception if the descriptor is null.
template <typename Child_descriptor>
inline std::shared_ptr<Child_descriptor> unpack_descriptor(
    const hipdnnBackendDescriptor* descriptor, hipdnnStatus_t status, const std::string& message)
{
    static_assert(is_valid_backend_child_descriptor_v<Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnBackendDescriptorImpl");

    THROW_IF_NULL(descriptor, status, message);
    THROW_IF_NULL(descriptor->impl, status, message);
    THROW_IF_NE(descriptor->get_type(),
                Child_descriptor::get_static_type(),
                hipdnnStatus_t::HIPDNN_STATUS_BAD_PARAM,
                "Unpacking hipdnnBackendDescriptor failed: Descriptor type mismatch.");

    auto child_descriptor
        = std::static_pointer_cast<Child_descriptor>(descriptor->impl);

    return child_descriptor;
}

// Unpacks a hipdnnBackendDescriptor from an array of elements.
// The array of elements is expected to be a pointer to a hipdnnBackendDescriptor.
// Throws an exception if the array of elements is null or if the descriptor cannot be unpacked.
template <typename Child_descriptor>
inline std::shared_ptr<Child_descriptor> unpack_descriptor(const void* array_of_elements,
                                                           hipdnnStatus_t status,
                                                           const std::string& message)
{
    static_assert(is_valid_backend_child_descriptor_v<Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnBackendDescriptorImpl");

    THROW_IF_NULL(array_of_elements, status, message);

    return unpack_descriptor<Child_descriptor>(
        *static_cast<hipdnnBackendDescriptor* const*>(array_of_elements), status, message);
}

// Packs a shared_ptr into a new hipdnnBackendDescriptor.
// Note: Ownership of the descriptor is transferred to the caller,
//       and it is the caller's responsibility to delete it.
//       You can use Scoped_descriptor to manage the lifetime of the descriptor automatically.
inline hipdnnBackendDescriptor*
    pack_descriptor(const std::shared_ptr<const Backend_descriptor_interface>& impl)
{
    auto descriptor = new hipdnnBackendDescriptor();
    descriptor->impl = std::const_pointer_cast<Backend_descriptor_interface>(impl);
    return descriptor;
}

// Packs a shared_ptr into a new hipdnnBackendDescriptor, and stores it into a provided array_of_elements ptr.
// Note: Ownership of the descriptor is transferred to the public API caller.
//       It is the caller's responsibility to delete it.
inline void pack_descriptor(const std::shared_ptr<const Backend_descriptor_interface>& impl,
                            void*& array_of_elements)
{
    *static_cast<hipdnnBackendDescriptor**>(array_of_elements)
        = pack_descriptor(impl);
}
}
//NOLINTEND(readability-identifier-naming)
