// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptor_factory.hpp"
#include "engine_config_descriptor.hpp"
#include "engine_descriptor.hpp"
#include "engine_heuristic_descriptor.hpp"
#include "error.hpp"
#include "execution_plan_descriptor.hpp"
#include "graph_descriptor.hpp"
#include "hipdnn_exception.hpp"
#include "logging/logging.hpp"
#include "variant_descriptor.hpp"

namespace hipdnn_backend
{

void Descriptor_factory::create(hipdnnBackendDescriptorType_t descriptor_type,
                                hipdnnBackendDescriptor_t* descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");

    HIPDNN_LOG_INFO("Creating descriptor of type: {}",
                    hipdnn_get_backend_descriptor_type_name(descriptor_type));

    std::shared_ptr<Backend_descriptor_interface> private_desc;
    switch(descriptor_type)
    {
    case HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        private_desc = std::make_shared<Engine_config_descriptor>();
        break;
    case HIPDNN_BACKEND_ENGINE_DESCRIPTOR:
        private_desc = std::make_shared<Engine_descriptor>();
        break;
    case HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        private_desc = std::make_shared<Execution_plan_descriptor>();
        break;
    case HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        private_desc = std::make_shared<Graph_descriptor>();
        break;
    case HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        private_desc = std::make_shared<Variant_descriptor>();
        break;
    case HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        private_desc = std::make_shared<Engine_heuristic_descriptor>();
        break;
    default:
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                               std::string("Descriptor type ")
                                   + hipdnn_get_backend_descriptor_type_name(descriptor_type)
                                   + " is not supported.");
    }

    *descriptor = hipdnnBackendDescriptor::pack_descriptor(private_desc);

    HIPDNN_LOG_INFO("Created descriptor: {:p}", static_cast<void*>(*descriptor));
}

void Descriptor_factory::create_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                          const uint8_t* serialized_graph,
                                          size_t graph_byte_size)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");
    THROW_IF_NULL(
        serialized_graph, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "serialized_graph is null.");
    THROW_IF_TRUE(graph_byte_size == 0, HIPDNN_STATUS_BAD_PARAM, "graph_byte_size is 0.");

    auto graph_descriptor = std::make_shared<Graph_descriptor>();
    graph_descriptor->deserialize_graph(serialized_graph, graph_byte_size);
    *descriptor = hipdnnBackendDescriptor::pack_descriptor(graph_descriptor);

    HIPDNN_LOG_INFO("Created graph descriptor: {:p}", static_cast<void*>(*descriptor));
}

void Descriptor_factory::destroy(hipdnnBackendDescriptor_t descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t is null.");

    delete descriptor;

    HIPDNN_LOG_INFO("Destroyed descriptor: {:p}", static_cast<void*>(descriptor));
}

} // namespace hipdnn_backend
