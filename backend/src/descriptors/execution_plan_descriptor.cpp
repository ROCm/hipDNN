// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "execution_plan_descriptor.hpp"
#include "error.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

Execution_plan_descriptor::Execution_plan_descriptor()
{
    type = HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
}

void Execution_plan_descriptor::finalize()
{
    if(is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Execution_plan_descriptor::finalize() failed: "
                               "Already finalized.");
    }

    if(_handle == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Execution_plan_descriptor::finalize() failed: "
                               "Handle was not set.");
    }

    if(_engine_config == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Execution_plan_descriptor::finalize() failed: "
                               "Engine was not set.");
    }

    hipdnnBackendDescriptor::finalize();
}

hipdnnStatus_t Execution_plan_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                        hipdnnBackendAttributeType_t attribute_type,
                                                        int64_t requested_element_count,
                                                        int64_t* element_count,
                                                        void* array_of_elements)
{
    if(!is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_NOT_INITIALIZED,
                              "Execution_plan_descriptor::get_attribute() failed: "
                              "Not finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        return get_workspace_size(
            attribute_type, requested_element_count, element_count, array_of_elements);
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    default:
        return set_last_error(
            HIPDNN_STATUS_NOT_SUPPORTED,
            (std::string(
                 "Execution_plan_descriptor::get_attribute() is not supported for attribute ")
             + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".")
                .c_str());
    }
}

hipdnnStatus_t
    Execution_plan_descriptor::get_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                                                  int64_t requested_element_count,
                                                  int64_t* element_count,
                                                  void* array_of_elements)
{
    if(_engine_config == nullptr)
    {
        // This would be a bug since we are finalized at this point
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Execution_plan_descriptor failed to get workspace size: "
                              "Engine was not set (internal error).");
    }

    return _engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                         attribute_type,
                                         requested_element_count,
                                         element_count,
                                         array_of_elements);
}

hipdnnStatus_t Execution_plan_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                        hipdnnBackendAttributeType_t attribute_type,
                                                        int64_t element_count,
                                                        const void* array_of_elements)
{
    if(is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_NOT_INITIALIZED,
                              "Execution_plan_descriptor::set_attribute() failed: "
                              "Already finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
        return set_handle(attribute_type, element_count, array_of_elements);
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        return set_engine_config(attribute_type, element_count, array_of_elements);
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    default:
        return set_last_error(
            HIPDNN_STATUS_NOT_SUPPORTED,
            (std::string(
                 "Execution_plan_descriptor::set_attribute() is not supported for attribute ")
             + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".")
                .c_str());
    }
}

hipdnnStatus_t Execution_plan_descriptor::set_handle(hipdnnBackendAttributeType_t attribute_type,
                                                     int64_t element_count,
                                                     const void* array_of_elements)
{
    if(attribute_type != HIPDNN_TYPE_HANDLE)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM,
                              "Execution_plan_descriptor failed to set handle: "
                              "Invalid attribute type.");
    }
    if(element_count != 1)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM,
                              "Execution_plan_descriptor failed to set handle: "
                              "Invalid element count.");
    }
    if(array_of_elements == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Execution_plan_descriptor failed to set handle: "
                              "Null pointer.");
    }

    hipdnnHandle_t handle = *reinterpret_cast<const hipdnnHandle_t*>(array_of_elements);

    if(handle == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Execution_plan_descriptor failed to set handle: "
                              "Handle is null.");
    }

    _handle = handle;
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
    Execution_plan_descriptor::set_engine_config(hipdnnBackendAttributeType_t attribute_type,
                                                 int64_t element_count,
                                                 const void* array_of_elements)
{
    if(attribute_type != HIPDNN_TYPE_BACKEND_DESCRIPTOR)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Invalid attribute type.");
    }

    if(element_count != 1)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Invalid element count.");
    }

    if(array_of_elements == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Null pointer.");
    }

    hipdnnBackendDescriptor_t engine_config
        = *reinterpret_cast<const hipdnnBackendDescriptor_t*>(array_of_elements);

    if(engine_config == nullptr)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Engine config is null.");
    }

    if(engine_config->type != HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Invalid engine config descriptor type.");
    }

    if(!engine_config->is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                              "Execution_plan_descriptor failed to set engine config: "
                              "Engine config descriptor is not finalized.");
    }

    _engine_config = engine_config;
    return HIPDNN_STATUS_SUCCESS;
}

} // namespace hipdnn_backend
