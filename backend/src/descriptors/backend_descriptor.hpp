// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include "hipdnn_exception.hpp"
#include <memory>

struct Backend_descriptor_interface
{
    virtual ~Backend_descriptor_interface() = default;

    virtual void finalize() = 0;
    virtual bool is_finalized() const = 0;
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

    virtual hipdnnBackendDescriptorType_t get_type() const = 0;
};

//NOLINTBEGIN(readability-identifier-naming)
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

    void finalize() override
    {
        _finalized = true;
    }
    bool is_finalized() const override
    {
        return _finalized;
    }

    hipdnnBackendDescriptorType_t get_type() const override
    {
        return _type;
    }

    static hipdnnBackendDescriptorType_t get_static_type()
    {
        return T::get_static_type();
    }
};
}

namespace
{
// Helper trait for static_asserts in unpack.
template <typename Child_descriptor>
constexpr bool is_valid_backend_child_descriptor_v = std::is_base_of_v<
    hipdnn_backend::hipdnnBackendDescriptorImpl<std::remove_const_t<Child_descriptor>>,
    std::remove_const_t<Child_descriptor>>;
}

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

    template <typename Child_descriptor>
    std::shared_ptr<Child_descriptor> as_descriptor() const
    {
        return unpack_descriptor<Child_descriptor>(
            this,
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Failed to cast backend descriptor: Null descriptor provided.");
    }

    bool is_valid();

    hipdnnBackendDescriptorType_t get_type() const override;

    bool operator==(const hipdnnBackendDescriptor& other) const;

    // Unpacks a hipdnnBackendDescriptor into a shared_ptr of the specified type.
    // Throws an exception if the descriptor is null.
    template <typename Child_descriptor>
    static std::shared_ptr<Child_descriptor>
        unpack_descriptor(const hipdnnBackendDescriptor* descriptor,
                          hipdnnStatus_t status,
                          const std::string& message)
    {
        static_assert(is_valid_backend_child_descriptor_v<Child_descriptor>,
                      "Child_descriptor must inherit from hipdnnBackendDescriptorImpl");

        THROW_IF_NULL(descriptor, status, message);
        THROW_IF_NULL(descriptor->_impl, status, message);
        THROW_IF_NE(descriptor->get_type(),
                    Child_descriptor::get_static_type(),
                    hipdnnStatus_t::HIPDNN_STATUS_BAD_PARAM,
                    "Unpacking hipdnnBackendDescriptor failed: Descriptor type mismatch.");

        auto child_descriptor = std::static_pointer_cast<Child_descriptor>(descriptor->_impl);

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
    static hipdnnBackendDescriptor*
        pack_descriptor(const std::shared_ptr<const Backend_descriptor_interface>& _impl)
    {
        auto descriptor = new hipdnnBackendDescriptor();
        descriptor->_impl = std::const_pointer_cast<Backend_descriptor_interface>(_impl);
        return descriptor;
    }

    // Packs a shared_ptr into a new hipdnnBackendDescriptor, and stores it into a provided array_of_elements ptr.
    // Note: Ownership of the descriptor is transferred to the public API caller.
    //       It is the caller's responsibility to delete it.
    static void pack_descriptor(const std::shared_ptr<const Backend_descriptor_interface>& _impl,
                                void*& array_of_elements)
    {
        *static_cast<hipdnnBackendDescriptor**>(array_of_elements) = pack_descriptor(_impl);
    }

private:
    std::shared_ptr<Backend_descriptor_interface> _impl;
};
//NOLINTEND(readability-identifier-naming)
