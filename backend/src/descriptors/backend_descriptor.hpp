// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include <memory>

//NOLINTBEGIN(readability-identifier-naming)
struct hipdnnPrivateBackendDescriptor;

template <typename Child_descriptor>
static std::shared_ptr<Child_descriptor>
    unpack_descriptor(const hipdnnBackendDescriptor* descriptor,
                      hipdnnStatus_t error_status,
                      const std::string& error_message);

struct hipdnnBackendDescriptor
{
    std::shared_ptr<hipdnnPrivateBackendDescriptor> private_descriptor;

    template <typename Child_descriptor>
    std::shared_ptr<Child_descriptor> as_descriptor() const
    {
        return unpack_descriptor<Child_descriptor>(
            this,
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Failed to cast backend descriptor: Null descriptor provided.");
    }
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
};

struct Scoped_descriptor
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    Scoped_descriptor() = default;
    explicit Scoped_descriptor(hipdnnBackendDescriptor_t desc)
        : descriptor(desc)
    {
    }
    ~Scoped_descriptor()
    {
        delete descriptor;
    }
    Scoped_descriptor(const Scoped_descriptor&) = delete;
    Scoped_descriptor& operator=(const Scoped_descriptor&) = delete;
    Scoped_descriptor(Scoped_descriptor&& other) noexcept
    {
        descriptor = other.descriptor;
        other.descriptor = nullptr;
    }
    Scoped_descriptor& operator=(Scoped_descriptor&& other) noexcept
    {
        if(this != &other)
        {
            delete descriptor;
            descriptor = other.descriptor;
            other.descriptor = nullptr;
        }
        return *this;
    }
    hipdnnBackendDescriptor_t get() const
    {
        return descriptor;
    }
    hipdnnBackendDescriptor_t* get_ptr()
    {
        return &descriptor;
    }
};

// Unpacks a hipdnnBackendDescriptor into a shared_ptr of the specified type.
// Throws an exception if the descriptor is null.
template <typename Child_descriptor>
static std::shared_ptr<Child_descriptor> unpack_descriptor(
    const hipdnnBackendDescriptor* descriptor, hipdnnStatus_t status, const std::string& message)
{
    static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

    THROW_IF_NULL(descriptor, status, message);
    THROW_IF_NULL(descriptor->private_descriptor, status, message);

    auto child_descriptor
        = std::static_pointer_cast<Child_descriptor>(descriptor->private_descriptor);

    return child_descriptor;
}

// Unpacks a hipdnnBackendDescriptor from an array of elements.
// The array of elements is expected to be a pointer to a hipdnnBackendDescriptor.
// Throws an exception if the array of elements is null or if the descriptor cannot be unpacked.
template <typename Child_descriptor>
static std::shared_ptr<Child_descriptor> unpack_descriptor(const void* array_of_elements,
                                                           hipdnnStatus_t status,
                                                           const std::string& message)
{
    static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

    THROW_IF_NULL(array_of_elements, status, message);

    return unpack_descriptor<Child_descriptor>(
        *static_cast<hipdnnBackendDescriptor* const*>(array_of_elements), status, message);
}

// Packs a shared_ptr into a new hipdnnBackendDescriptor.
// Note: Ownership of the descriptor is transferred to the caller,
//       and it is the caller's responsibility to delete it.
//       You can use Scoped_descriptor to manage the lifetime of the descriptor automatically.
template <typename Child_descriptor>
static hipdnnBackendDescriptor*
    pack_descriptor(const std::shared_ptr<const Child_descriptor>& private_descriptor)
{
    static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

    auto descriptor = new hipdnnBackendDescriptor();
    descriptor->private_descriptor = std::const_pointer_cast<hipdnnPrivateBackendDescriptor>(
        std::static_pointer_cast<const hipdnnPrivateBackendDescriptor>(private_descriptor));
    return descriptor;
}

// Packs a shared_ptr into a new hipdnnBackendDescriptor.
// Note: Ownership of the descriptor is transferred to the caller,
//       and it is the caller's responsibility to delete it.
//       You can use Scoped_descriptor to manage the lifetime of the descriptor automatically.
template <typename Child_descriptor>
static hipdnnBackendDescriptor*
    pack_descriptor(const std::shared_ptr<Child_descriptor>& private_descriptor)
{
    static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

    auto descriptor = new hipdnnBackendDescriptor();
    descriptor->private_descriptor
        = std::static_pointer_cast<hipdnnPrivateBackendDescriptor>(private_descriptor);
    return descriptor;
}

// Packs a shared_ptr into a new hipdnnBackendDescriptor, and stores it into a provided array_of_elements ptr.
// Note: Ownership of the descriptor is transferred to the public API caller.
//       It is the caller's responsibility to delete it.
template <typename Child_descriptor>
static void pack_descriptor(const std::shared_ptr<const Child_descriptor>& private_descriptor,
                            void*& array_of_elements)
{
    static_assert(std::is_base_of_v<hipdnnPrivateBackendDescriptor, Child_descriptor>,
                  "Child_descriptor must inherit from hipdnnPrivateBackendDescriptor");

    *static_cast<hipdnnBackendDescriptor**>(array_of_elements)
        = pack_descriptor(private_descriptor);
}
//NOLINTEND(readability-identifier-naming)