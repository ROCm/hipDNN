// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_config_descriptor.hpp"
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
    if(is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor::finalize() failed: "
                               "Already finalized.");
    }

    if(_engine == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor::finalize() failed: "
                               "Engine is not set.");
    }

    if(_max_workspace_size == INVALID_WORKSPACE_SIZE)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor::finalize() failed: "
                               "Max workspace size is not set.");
    }

    hipdnnBackendDescriptor::finalize();
}

hipdnnStatus_t Engine_config_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                       hipdnnBackendAttributeType_t attribute_type,
                                                       int64_t requested_element_count,
                                                       int64_t* element_count,
                                                       void* array_of_elements)
{
    if(!is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_INITIALIZED,
                               "Engine_config_descriptor::get_attribute() failed: "
                               "Not finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        get_engine(attribute_type, requested_element_count, element_count, array_of_elements);
        return HIPDNN_STATUS_SUCCESS;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
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
    if(attribute_type != HIPDNN_TYPE_BACKEND_DESCRIPTOR)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor failed to get engine: "
                               "Invalid attribute type.");
    }

    if(requested_element_count != 1)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor failed to get engine: "
                               "Invalid element count.");
    }

    if(array_of_elements == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "Engine_config_descriptor failed to get engine: "
                               "Null pointer.");
    }

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    *reinterpret_cast<hipdnnBackendDescriptor_t*>(array_of_elements) = _engine;
}

hipdnnStatus_t Engine_config_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                       hipdnnBackendAttributeType_t attribute_type,
                                                       int64_t element_count,
                                                       const void* array_of_elements)
{
    if(is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_INITIALIZED,
                               "Engine_config_descriptor::set_attribute() failed: "
                               "Already finalized.");
    }

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
    if(attribute_type != HIPDNN_TYPE_BACKEND_DESCRIPTOR)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor failed to set engine: "
                               "Invalid attribute type.");
    }

    if(element_count != 1)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor failed to set engine: "
                               "Invalid element count.");
    }

    if(array_of_elements == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "Engine_config_descriptor failed to set engine: "
                               "Null pointer.");
    }

    hipdnnBackendDescriptor_t engine
        = *reinterpret_cast<const hipdnnBackendDescriptor_t*>(array_of_elements);

    if(engine == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "Engine_config_descriptor failed to set engine: "
                               "Engine is null.");
    }

    if(engine->type != HIPDNN_BACKEND_ENGINE_DESCRIPTOR)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_config_descriptor failed to set engine: "
                               "Invalid engine descriptor type.");
    }

    if(!engine->is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                               "Engine_config_descriptor failed to set engine: "
                               "Engine is not finalized.");
    }

    _engine = engine;
}

hipdnnStatus_t Engine_config_descriptor::set_max_workspace_size(int64_t workspace_size)
{
    // This should only be called from the plugin manager, so all errors should be
    // internal errors rather than user errors.

    if(is_finalized())
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Internal error:  Failed to set max workspace size:"
                              "Already finalized.");
    }

    if(workspace_size < 0)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Internal error:  Failed to set max workspace size: "
                              "Max workspace size cannot be negative.");
    }

    _max_workspace_size = workspace_size;
    return HIPDNN_STATUS_SUCCESS;
}

} // namespace hipdnn_backend
