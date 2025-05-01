// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "variant_descriptor.hpp"
#include "error.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend_descriptor_type.h"

namespace hipdnn_backend
{

Variant_descriptor::Variant_descriptor()
{
    type = HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
}

hipdnnStatus_t Variant_descriptor::finalize()
{
    if(_data_pointers.empty() || _unique_ids.empty())
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    if(_data_pointers.size() != _unique_ids.size())
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }
    return Backend_descriptor::finalize();
}

hipdnnStatus_t Variant_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                 hipdnnBackendAttributeType_t attribute_type,
                                                 int64_t requested_element_count,
                                                 int64_t* element_count,
                                                 void* array_of_elements)
{
    if(element_count == nullptr || array_of_elements == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    if(!is_finalized())
    {
        return HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED;
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        if(attribute_type != HIPDNN_TYPE_VOID_PTR)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        *element_count = std::min<int64_t>(requested_element_count,
                                           static_cast<int64_t>(_data_pointers.size()));
        for(size_t i = 0; i < static_cast<size_t>(*element_count); ++i)
        {
            static_cast<void**>(array_of_elements)[i] = const_cast<void*>(_data_pointers[i]);
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        if(attribute_type != HIPDNN_TYPE_INT64)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        *element_count
            = std::min<int64_t>(requested_element_count, static_cast<int64_t>(_unique_ids.size()));
        for(size_t i = 0; i < static_cast<size_t>(*element_count); ++i)
        {
            static_cast<int64_t*>(array_of_elements)[i] = _unique_ids[i];
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        if(attribute_type != HIPDNN_TYPE_VOID_PTR || requested_element_count != 1)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        *element_count = 1;
        *static_cast<void**>(array_of_elements) = _workspace;
        break;

    default:
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t Variant_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                 hipdnnBackendAttributeType_t attribute_type,
                                                 int64_t element_count,
                                                 const void* array_of_elements)
{
    if(is_finalized())
    {
        return HIPDNN_STATUS_NOT_INITIALIZED;
    }

    if(array_of_elements == nullptr || element_count <= 0)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        if(attribute_type != HIPDNN_TYPE_VOID_PTR)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        _data_pointers.assign(static_cast<const void* const*>(array_of_elements),
                              static_cast<const void* const*>(array_of_elements) + element_count);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        if(attribute_type != HIPDNN_TYPE_INT64)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        _unique_ids.assign(static_cast<const int64_t*>(array_of_elements),
                           static_cast<const int64_t*>(array_of_elements) + element_count);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        if(attribute_type != HIPDNN_TYPE_VOID_PTR || element_count != 1)
        {
            return HIPDNN_STATUS_BAD_PARAM;
        }
        _workspace = *static_cast<void* const*>(array_of_elements);
        break;

    default:
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return HIPDNN_STATUS_SUCCESS;
}

}
