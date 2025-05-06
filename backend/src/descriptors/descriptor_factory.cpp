// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptor_factory.hpp"
#include "error.hpp"
#include "execution_plan_descriptor.hpp"
#include "graph_descriptor.hpp"
#include "variant_descriptor.hpp"
#include <hipdnn_sdk/logging/logger.hpp>

namespace hipdnn_backend
{

hipdnnStatus_t Descriptor_factory::create(hipdnnBackendDescriptorType_t descriptor_type,
                                          hipdnnBackendDescriptor_t* descriptor)
{
    if(descriptor == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    HIPDNN_LOG_INFO("Creating descriptor of type: {}",
                    hipdnn_get_backend_descriptor_type_name(descriptor_type));

    switch(descriptor_type)
    {
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
        return set_last_error(HIPDNN_STATUS_NOT_SUPPORTED,
                              (std::string("Descriptor type ")
                               + hipdnn_get_backend_descriptor_type_name(descriptor_type)
                               + " is not supported.")
                                  .c_str());
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t Descriptor_factory::create_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                                    const uint8_t* serialized_graph,
                                                    size_t graph_byte_size)
{
    if(descriptor == nullptr || serialized_graph == nullptr || graph_byte_size == 0)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    auto graph_descriptor = new Graph_descriptor();
    if(auto status = graph_descriptor->deserialize_graph(serialized_graph, graph_byte_size);
       status != HIPDNN_STATUS_SUCCESS)
    {
        delete graph_descriptor;
        return status;
    }
    *descriptor = graph_descriptor;

    return HIPDNN_STATUS_SUCCESS;
}

} // namespace hipdnn_backend
