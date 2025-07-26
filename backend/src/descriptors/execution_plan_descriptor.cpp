// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "execution_plan_descriptor.hpp"
#include "engine_config_descriptor.hpp"
#include "error.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

void Execution_plan_descriptor::finalize()
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Execution_plan_descriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_handle,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Execution_plan_descriptor::finalize() failed: Handle was not set.");

    THROW_IF_NULL(_engine_config,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Execution_plan_descriptor::finalize() failed: Engine was not set.");

    hipdnnBackendDescriptorImpl<Execution_plan_descriptor>::finalize();
}

void Execution_plan_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                              hipdnnBackendAttributeType_t attribute_type,
                                              int64_t requested_element_count,
                                              int64_t* element_count,
                                              void* array_of_elements) const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Execution_plan_descriptor::get_attribute() failed: Not finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        get_workspace_size(
            attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        get_engine_config(
            attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "Execution_plan_descriptor::get_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Execution_plan_descriptor::get_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                                                   int64_t requested_element_count,
                                                   int64_t* element_count,
                                                   void* array_of_elements) const
{
    THROW_IF_NULL(_engine_config,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "Execution_plan_descriptor failed to get workspace size: Engine was not set "
                  "(internal error).");

    _engine_config->get_attribute(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                  attribute_type,
                                  requested_element_count,
                                  element_count,
                                  array_of_elements);
}

void Execution_plan_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                              hipdnnBackendAttributeType_t attribute_type,
                                              int64_t element_count,
                                              const void* array_of_elements)
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "Execution_plan_descriptor::set_attribute() failed: Already finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
        set_handle(attribute_type, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        set_engine_config(attribute_type, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "Execution_plan_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Execution_plan_descriptor::set_handle(hipdnnBackendAttributeType_t attribute_type,
                                           int64_t element_count,
                                           const void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_HANDLE,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to set handle: Invalid attribute type.");
    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to set handle: Invalid element count.");
    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Execution_plan_descriptor failed to set handle: Null pointer.");

    hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(array_of_elements);

    THROW_IF_NULL(handle,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Execution_plan_descriptor failed to set handle: Handle is null.");

    _handle = handle;
}

void Execution_plan_descriptor::set_engine_config(hipdnnBackendAttributeType_t attribute_type,
                                                  int64_t element_count,
                                                  const void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to set engine config: Invalid attribute type.");

    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to set engine config: Invalid element count.");

    auto engine_config = hipdnnBackendDescriptor::unpack_descriptor<const Engine_config_descriptor>(
        array_of_elements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "Execution_plan_descriptor failed to set engine config: Null pointer.");

    THROW_IF_FALSE(engine_config->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Execution_plan_descriptor failed to set engine config: Engine config "
                   "descriptor is not finalized.");

    _engine_config = engine_config;
}

void Execution_plan_descriptor::get_engine_config(hipdnnBackendAttributeType_t attribute_type,
                                                  int64_t requested_element_count,
                                                  int64_t* element_count,
                                                  void* array_of_elements) const
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to get engine config: Invalid attribute type.");

    THROW_IF_NE(requested_element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Execution_plan_descriptor failed to get engine config: "
                "Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Execution_plan_descriptor failed to get engine config: Null pointer for "
                  "array_of_elements.");

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    THROW_IF_NULL(_engine_config,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Execution_plan_descriptor failed to get engine config: Engine config is null. "
                  "Engine config was not set.");

    hipdnnBackendDescriptor::pack_descriptor(_engine_config, array_of_elements);
}

std::shared_ptr<const Engine_config_descriptor> Execution_plan_descriptor::get_engine_config() const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "Execution_plan_descriptor::get_engine_config() failed: Not finalized.");

    return _engine_config;
}

hipdnnBackendDescriptorType_t Execution_plan_descriptor::get_static_type()
{
    return HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
}

} // namespace hipdnn_backend
