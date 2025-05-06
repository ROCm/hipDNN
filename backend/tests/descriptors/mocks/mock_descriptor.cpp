// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "mock_descriptor.hpp"
#include "error.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

#include <utility>

namespace hipdnn_backend
{

Mock_descriptor::Mock_descriptor(hipdnnBackendDescriptorType_t desc_type, bool finalized)
    : hipdnnBackendDescriptor()
{
    type = desc_type;
    if(finalized)
    {
        hipdnnBackendDescriptor::finalize();
    }
}

hipdnnStatus_t Mock_descriptor::set_data(hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t element_count,
                                         const void* array_of_elements)
{
    if(element_count <= 0)
    {
        return set_last_error(
            HIPDNN_STATUS_BAD_PARAM,
            "Mock_descriptor::set_attribute() called with invalid element count.");
    }
    if(array_of_elements == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Mock_descriptor::set_attribute() called with null elements.");
    }

    std::vector<const void*> elements;
    elements.assign(static_cast<const void* const*>(array_of_elements),
                    static_cast<const void* const*>(array_of_elements) + element_count);

    auto key = std::make_pair(attribute_name, attribute_type);
    _attributes[key] = elements;
    return HIPDNN_STATUS_SUCCESS;
}

void Mock_descriptor::finalize()
{
    if(is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_INITIALIZED,
                               "Mock_descriptor::finalize() is already finalized.");
    }

    hipdnnBackendDescriptor::finalize();
}

hipdnnStatus_t Mock_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                              hipdnnBackendAttributeType_t attribute_type,
                                              int64_t requested_element_count,
                                              int64_t* element_count,
                                              void* array_of_elements)
{
    auto key = std::make_pair(attribute_name, attribute_type);
    if(_attributes.find(key) == _attributes.end())
    {
        return set_last_error(HIPDNN_STATUS_NOT_SUPPORTED,
                              "Mock_descriptor::get_attribute() called but not set.");
    }

    if(requested_element_count <= 0)
    {
        return set_last_error(
            HIPDNN_STATUS_BAD_PARAM,
            "Mock_descriptor::get_attribute() called with invalid element count.");
    }

    if(array_of_elements == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Mock_descriptor::get_attribute() called with null elements.");
    }

    int64_t count
        = std::min<int64_t>(requested_element_count, static_cast<int64_t>(_attributes[key].size()));
    if(element_count != nullptr)
    {
        *element_count = count;
    }

    for(size_t i = 0; i < static_cast<size_t>(count); ++i)
    {
        static_cast<void**>(array_of_elements)[i] = const_cast<void*>(_attributes[key][i]);
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t Mock_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                              hipdnnBackendAttributeType_t attribute_type,
                                              int64_t element_count,
                                              const void* array_of_elements)
{
    // TODO can add some checks if needed including a way to mock errors for particular attrs
    return set_data(attribute_name, attribute_type, element_count, array_of_elements);
}

} // namespace hipdnn_backend
