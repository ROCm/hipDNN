// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "variant_descriptor.hpp"
#include "error.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

Variant_descriptor::Variant_descriptor()
{
    type = HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
}

void Variant_descriptor::finalize()
{
    THROW_IF_NE(_data_pointers.size(),
                _unique_ids.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "Data pointers and unique ids don't match");
    THROW_IF_TRUE(
        _data_pointers.empty(), HIPDNN_STATUS_BAD_PARAM, "Data pointers and unique ids are empty");

    hipdnnBackendDescriptor::finalize();
}

void Variant_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                       hipdnnBackendAttributeType_t attribute_type,
                                       int64_t requested_element_count,
                                       int64_t* element_count,
                                       void* array_of_elements)
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Variant_descriptor::get_attribute() failed: Not finalized.");
    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Variant_descriptor::get_attribute(): array_of_elements is null");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::get_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_VOID_PTR for DATA_POINTERS");
        THROW_IF_NULL(element_count,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Variant_descriptor::get_attribute(): element_count is null");
        *element_count = std::min<int64_t>(requested_element_count,
                                           static_cast<int64_t>(_data_pointers.size()));
        for(size_t i = 0; i < static_cast<size_t>(*element_count); ++i)
        {
            static_cast<void**>(array_of_elements)[i] = const_cast<void*>(_data_pointers[i]);
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::get_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_INT64 for UNIQUE_IDS");
        THROW_IF_NULL(element_count,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Variant_descriptor::get_attribute(): element_count is null");
        *element_count
            = std::min<int64_t>(requested_element_count, static_cast<int64_t>(_unique_ids.size()));
        for(size_t i = 0; i < static_cast<size_t>(*element_count); ++i)
        {
            static_cast<int64_t*>(array_of_elements)[i] = _unique_ids[i];
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::get_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_VOID_PTR for WORKSPACE");
        THROW_IF_FALSE(requested_element_count == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::get_attribute(): requested_element_count "
                       "is not 1 for WORKSPACE");
        if(element_count != nullptr)
        {
            *element_count = 1;
        }

        *static_cast<void**>(array_of_elements) = _workspace;
        break;

    default:
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                               "Variant_descriptor::get_attribute: attribute_name not supported");
    }
}

void Variant_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                       hipdnnBackendAttributeType_t attribute_type,
                                       int64_t element_count,
                                       const void* array_of_elements)
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "Variant_descriptor::set_attribute() failed: Already finalized.");
    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Variant_descriptor::set_attribute(): array_of_elements is null");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::set_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_VOID_PTR for DATA_POINTERS");
        _data_pointers.assign(static_cast<const void* const*>(array_of_elements),
                              static_cast<const void* const*>(array_of_elements) + element_count);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_INT64,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::set_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_INT64 for UNIQUE_IDS");
        _unique_ids.assign(static_cast<const int64_t*>(array_of_elements),
                           static_cast<const int64_t*>(array_of_elements) + element_count);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        THROW_IF_FALSE(attribute_type == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::set_attribute(): attribute_type is not "
                       "HIPDNN_TYPE_VOID_PTR for WORKSPACE");
        THROW_IF_FALSE(element_count == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Variant_descriptor::set_attribute(): element_count is not 1 for WORKSPACE");

        _workspace = *static_cast<void* const*>(array_of_elements);
        break;

    default:
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                               "Variant_descriptor::set_attribute: attribute_name not supported");
    }
}

}
