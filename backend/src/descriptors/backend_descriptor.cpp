// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "backend_descriptor.hpp"
#include "error.hpp"

using namespace hipdnn_backend;

void hipdnnBackendDescriptor::finalize()
{
    THROW_IF_TRUE(!impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null impl in finalize.");
    impl->finalize();
}

bool hipdnnBackendDescriptor::is_finalized() const
{
    THROW_IF_TRUE(!impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null impl in is_finalized.");
    return impl->is_finalized();
}

void hipdnnBackendDescriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                            hipdnnBackendAttributeType_t attribute_type,
                                            int64_t requested_element_count,
                                            int64_t* element_count,
                                            void* array_of_elements) const
{
    THROW_IF_TRUE(!impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null impl in get_attribute.");
    impl->get_attribute(attribute_name, attribute_type, requested_element_count, element_count, array_of_elements);
}

void hipdnnBackendDescriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                            hipdnnBackendAttributeType_t attribute_type,
                                            int64_t element_count,
                                            const void* array_of_elements)
{
    THROW_IF_TRUE(!impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null impl in set_attribute.");
    impl->set_attribute(attribute_name, attribute_type, element_count, array_of_elements);
}

hipdnnBackendDescriptorType_t hipdnnBackendDescriptor::get_type() const
{
    THROW_IF_TRUE(!impl, HIPDNN_STATUS_INTERNAL_ERROR, "Null impl in get_type.");
    return impl->get_type();
}
