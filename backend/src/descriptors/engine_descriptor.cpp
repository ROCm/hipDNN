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
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_INITIALIZED,
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
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Engine_descriptor::get_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

hipdnnStatus_t Engine_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                hipdnnBackendAttributeType_t attribute_type,
                                                int64_t element_count,
                                                const void* array_of_elements)
{
    if(is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_INITIALIZED,
                               "Engine_descriptor::set_attribute() failed: "
                               "Already finalized.");
    }

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
        set_graph(attribute_type, element_count, array_of_elements);
        return HIPDNN_STATUS_SUCCESS;
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Engine_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Engine_descriptor::set_graph(hipdnnBackendAttributeType_t attribute_type,
                                  int64_t element_count,
                                  const void* array_of_elements)
{
    if(attribute_type != HIPDNN_TYPE_BACKEND_DESCRIPTOR)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_descriptor failed to set graph: "
                               "Invalid attribute type.");
    }

    if(element_count != 1)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_descriptor failed to set graph: "
                               "Invalid element count.");
    }

    if(array_of_elements == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "Engine_descriptor failed to set graph: "
                               "Null pointer.");
    }

    hipdnnBackendDescriptor_t graph
        = *reinterpret_cast<const hipdnnBackendDescriptor_t*>(array_of_elements);

    if(graph == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "Engine_descriptor failed to set graph: "
                               "Graph is null.");
    }

    if(graph->type != HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_descriptor failed to set graph: "
                               "Invalid engine descriptor type.");
    }

    if(!graph->is_finalized())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                               "Engine_descriptor failed to set graph: "
                               "Graph  is not finalized.");
    }

    _graph = graph;
}

} // namespace hipdnn_backend
