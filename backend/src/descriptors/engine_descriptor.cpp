// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_descriptor.hpp"
#include "error.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

Engine_descriptor::Engine_descriptor()
{
    type = HIPDNN_BACKEND_ENGINE_DESCRIPTOR;
}

void Engine_descriptor::finalize()
{
    throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                           "Engine_descriptor::finalize() is not implemented yet.");
    // TODO call base finalize();
}

hipdnnStatus_t
    Engine_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                     [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                     [[maybe_unused]] int64_t requested_element_count,
                                     [[maybe_unused]] int64_t* element_count,
                                     [[maybe_unused]] void* array_of_elements)
{
    if(!is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_NOT_INITIALIZED,
                              "Engine_descriptor::get_attribute() failed: "
                              "Not finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        return set_last_error(
            HIPDNN_STATUS_NOT_SUPPORTED,
            (std::string("Engine_descriptor::get_attribute() is not supported for attribute ")
             + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".")
                .c_str());
    }
}

hipdnnStatus_t
    Engine_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                     [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                     [[maybe_unused]] int64_t element_count,
                                     [[maybe_unused]] const void* array_of_elements)
{
    if(is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_NOT_INITIALIZED,
                              "Engine_descriptor::set_attribute() failed: "
                              "Already finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        return set_last_error(
            HIPDNN_STATUS_NOT_SUPPORTED,
            (std::string("Engine_descriptor::set_attribute() is not supported for attribute ")
             + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".")
                .c_str());
    }
}

} // namespace hipdnn_backend
