// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptor_factory.hpp"
#include "engine_config_descriptor.hpp"
#include "engine_descriptor.hpp"
#include "error.hpp"
#include "execution_plan_descriptor.hpp"
#include "graph_descriptor.hpp"
#include "hipdnn_exception.hpp"
#include "variant_descriptor.hpp"
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_backend
{

void Descriptor_factory::create(hipdnnBackendDescriptorType_t descriptor_type,
                                hipdnnBackendDescriptor_t* descriptor)
{
    if(descriptor == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "hipdnnBackendDescriptor_t* is null.");
    }

    HIPDNN_LOG_INFO("Creating descriptor of type: {}",
                    hipdnn_get_backend_descriptor_type_name(descriptor_type));

    switch(descriptor_type)
    {
    case HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        *descriptor = new Engine_config_descriptor();
        break;
    case HIPDNN_BACKEND_ENGINE_DESCRIPTOR:
        *descriptor = new Engine_descriptor();
        break;
    case HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        *descriptor = new Execution_plan_descriptor();
        break;
    case HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        *descriptor = new Graph_descriptor();
        break;
    case HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        *descriptor = new Variant_descriptor();
        break;
    default:
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                               std::string("Descriptor type ")
                                   + hipdnn_get_backend_descriptor_type_name(descriptor_type)
                                   + " is not supported.");
    }
}

void Descriptor_factory::create_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                          const uint8_t* serialized_graph,
                                          size_t graph_byte_size)
{
    if(descriptor == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                               "hipdnnBackendDescriptor_t* is null.");
    }

    if(serialized_graph == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "serialized_graph is null.");
    }
    if(graph_byte_size == 0)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM, "graph_byte_size is 0.");
    }

    auto graph_descriptor = new Graph_descriptor();
    try
    {
        graph_descriptor->deserialize_graph(serialized_graph, graph_byte_size);
    }
    catch(const std::exception& e)
    {
        delete graph_descriptor;
        throw;
    }

    *descriptor = graph_descriptor;
}

} // namespace hipdnn_backend
