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
    throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                           "Engine_config_descriptor::finalize() is not implemented yet.");
    // TODO return Backend_descriptor::finalize();
}

hipdnnStatus_t Engine_config_descriptor::get_attribute(
    hipdnnBackendAttributeName_t attribute_name,
    [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
    [[maybe_unused]] int64_t requested_element_count,
    [[maybe_unused]] int64_t* element_count,
    [[maybe_unused]] void* array_of_elements)
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

} // namespace hipdnn_backend
