// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_config_descriptor.hpp"
#include "engine_descriptor.hpp"
#include "error.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

Engine_config_descriptor::Engine_config_descriptor()
{
    type = HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
}

void Engine_config_descriptor::finalize()
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_config_descriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_engine,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_config_descriptor::finalize() failed: Engine is not set.");

    hipdnnBackendDescriptor::finalize();
}

hipdnnStatus_t Engine_config_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                       hipdnnBackendAttributeType_t attribute_type,
                                                       int64_t requested_element_count,
                                                       int64_t* element_count,
                                                       void* array_of_elements)
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Engine_config_descriptor::get_attribute() failed: Not finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        get_engine(attribute_type, requested_element_count, element_count, array_of_elements);
        return HIPDNN_STATUS_SUCCESS;
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
        get_max_workspace_size(
            attribute_type, requested_element_count, element_count, array_of_elements);
        return HIPDNN_STATUS_SUCCESS;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Engine_config_descriptor::get_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Engine_config_descriptor::get_engine(hipdnnBackendAttributeType_t attribute_type,
                                          int64_t requested_element_count,
                                          int64_t* element_count,
                                          void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_config_descriptor failed to get engine: "
                "Invalid attribute type.");

    THROW_IF_NE(requested_element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_config_descriptor failed to get engine: "
                "Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_config_descriptor failed to get engine: "
                  "Null pointer.");

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    pack_descriptor(_engine, array_of_elements);
}

void Engine_config_descriptor::get_max_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                                                      int64_t requested_element_count,
                                                      int64_t* element_count,
                                                      void* array_of_elements) const
{
    THROW_IF_NE(
        attribute_type,
        HIPDNN_TYPE_INT64,
        HIPDNN_STATUS_BAD_PARAM,
        "Engine_config_descriptor failed to get max workspace size: Invalid attribute type.");

    THROW_IF_NE(
        requested_element_count,
        1,
        HIPDNN_STATUS_BAD_PARAM,
        "Engine_config_descriptor failed to get max workspace size: Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_config_descriptor failed to get max workspace size: Null pointer.");

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    *static_cast<int64_t*>(array_of_elements) = _max_workspace_size;
}

hipdnnStatus_t Engine_config_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                       hipdnnBackendAttributeType_t attribute_type,
                                                       int64_t element_count,
                                                       const void* array_of_elements)
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "Engine_config_descriptor::set_attribute() failed: Already finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        set_engine(attribute_type, element_count, array_of_elements);
        return HIPDNN_STATUS_SUCCESS;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Engine_config_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Engine_config_descriptor::set_engine(hipdnnBackendAttributeType_t attribute_type,
                                          int64_t element_count,
                                          const void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_config_descriptor failed to set engine: "
                "Invalid attribute type.");

    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_config_descriptor failed to set engine: "
                "Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_config_descriptor failed to set engine: "
                  "Null pointer.");

    const Engine_descriptor* engine = *static_cast<Engine_descriptor* const*>(array_of_elements);

    THROW_IF_NULL(engine,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_config_descriptor failed to set engine: "
                  "Engine is null.");

    THROW_IF_NE(engine->type,
                HIPDNN_BACKEND_ENGINE_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_config_descriptor failed to set engine: "
                "Invalid engine descriptor type.");

    THROW_IF_FALSE(engine->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Engine_config_descriptor failed to set engine: "
                   "Engine is not finalized.");

    _engine = engine;
}

void Engine_config_descriptor::set_max_workspace_size(int64_t workspace_size)
{
    // This should only be called from the plugin manager, so all errors should be
    // internal errors rather than user errors.

    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "Engine_config_descriptor::set_max_workspace_size() failed: "
                   "Not finalized.");

    THROW_IF_LT(workspace_size,
                0,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "Engine_config_descriptor::set_max_workspace_size() failed: "
                "Max workspace size cannot be negative.");

    _max_workspace_size = workspace_size;
}

} // namespace hipdnn_backend
