// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_heuristic_descriptor.hpp"
#include "engine_config_descriptor.hpp"
#include "engine_descriptor.hpp"
#include "error.hpp"
#include "graph_descriptor.hpp"
#include "handle/handle.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"
#include "scoped_descriptor.hpp"

namespace hipdnn_backend
{

void Engine_heuristic_descriptor::finalize()
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_heuristic_descriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_graph,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_heuristic_descriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_heuristic_mode_set,
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_heuristic_descriptor::finalize() failed: Heuristic mode is not set.");

    auto handle = _graph->get_handle();
    auto plugin_resource_manager = handle->get_plugin_resource_manager();

    // TODO - For now we are going to return the engine IDs we get from the plugin resource manager.
    // In the future, we will need to implement a plugin system for engine heuristics that allows plugins to determine sort order of the returned engines.
    _engine_ids = plugin_resource_manager->get_applicable_engine_ids(_graph.get());

    hipdnnBackendDescriptorImpl<Engine_heuristic_descriptor>::finalize();
}

void Engine_heuristic_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                hipdnnBackendAttributeType_t attribute_type,
                                                int64_t requested_element_count,
                                                int64_t* element_count,
                                                void* array_of_elements) const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Engine_heuristic_descriptor::get_attribute() failed: Not finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        get_graph(attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        get_heuristic_mode(
            attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_RESULTS:
        get_engine_configs(
            attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "Engine_heuristic_descriptor::get_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Engine_heuristic_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                                hipdnnBackendAttributeType_t attribute_type,
                                                int64_t element_count,
                                                const void* array_of_elements)
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "Engine_heuristic_descriptor::set_attribute() failed: Already finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        set_heuristic_mode(attribute_type, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        set_graph(attribute_type, element_count, array_of_elements);
        break;
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string(
                "Engine_heuristic_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }
}

void Engine_heuristic_descriptor::set_heuristic_mode(hipdnnBackendAttributeType_t attribute_type,
                                                     int64_t element_count,
                                                     const void* array_of_elements)
{
    THROW_IF_NE(
        attribute_type,
        HIPDNN_TYPE_HEUR_MODE,
        HIPDNN_STATUS_BAD_PARAM,
        "Engine_heuristic_descriptor failed to set heuristic mode: Invalid attribute type.");

    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to set heuristic mode: Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_heuristic_descriptor failed to set heuristic mode: Null pointer.");

    auto heur_mode = static_cast<const hipdnnBackendHeurMode_t*>(array_of_elements);
    THROW_IF_NULL(
        heur_mode,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "Engine_heuristic_descriptor failed to set heuristic mode: Heuristic mode is null.");

    auto heur_mode_value = *heur_mode;
    if(heur_mode_value != HIPDNN_HEUR_MODE_FALLBACK)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                               "Engine_heuristic_descriptor::set_heuristic_mode() is not "
                               "supported for the given heuristic mode.");
    }

    _heuristic_mode = heur_mode_value;
    _heuristic_mode_set = true;
}

void Engine_heuristic_descriptor::set_graph(hipdnnBackendAttributeType_t attribute_type,
                                            int64_t element_count,
                                            const void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to set graph: Invalid element count.");

    auto graph = hipdnnBackendDescriptor::unpack_descriptor<const Graph_descriptor>(
        array_of_elements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "Engine_heuristic_descriptor failed to set graph: Null pointer.");

    THROW_IF_FALSE(graph->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Engine_heuristic_descriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void Engine_heuristic_descriptor::get_graph(hipdnnBackendAttributeType_t attribute_type,
                                            int64_t requested_element_count,
                                            int64_t* element_count,
                                            void* array_of_elements) const
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requested_element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_heuristic_descriptor failed to get graph: Null pointer.");

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    hipdnnBackendDescriptor::pack_descriptor(_graph, array_of_elements);
}

void Engine_heuristic_descriptor::get_engine_configs(hipdnnBackendAttributeType_t attribute_type,
                                                     int64_t requested_element_count,
                                                     int64_t* element_count,
                                                     void* array_of_elements) const
{
    THROW_IF_NE(
        attribute_type,
        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
        HIPDNN_STATUS_BAD_PARAM,
        "Engine_heuristic_descriptor failed to get engine configs: Invalid attribute type.");

    // Return the number of engine configs if they aren't requesting any.
    if(requested_element_count == 0)
    {
        THROW_IF_NULL(element_count,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Engine_heuristic_descriptor failed to get engine config count: Null pointer "
                      "for element count.");
        *element_count = static_cast<int64_t>(_engine_ids.size());
    }
    else
    {
        THROW_IF_NULL(array_of_elements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Engine_heuristic_descriptor failed to get engine configs: Null pointer.");

        THROW_IF_NULL(element_count,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Engine_heuristic_descriptor failed to get engine config: Null pointer for "
                      "element count.");

        // Create engine config descriptors for each engine ID
        auto output_array = static_cast<hipdnnBackendDescriptor_t*>(array_of_elements);
        for(size_t i = 0;
            std::cmp_less(i, _engine_ids.size()) && std::cmp_less(i, requested_element_count);
            ++i)
        {
            auto config = hipdnnBackendDescriptor::unpack_descriptor<Engine_config_descriptor>(
                output_array[i],
                HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                "Engine_heuristic_descriptor failed to get engine config: Config "
                "descriptor is null.");

            auto engine = std::make_shared<Engine_descriptor>();

            engine->set_attribute(
                HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &_engine_ids[i]);

            Scoped_descriptor graph_desc(hipdnnBackendDescriptor::pack_descriptor(_graph));
            engine->set_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  graph_desc.get_ptr());
            engine->finalize();

            Scoped_descriptor engine_desc(hipdnnBackendDescriptor::pack_descriptor(engine));
            config->set_attribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  engine_desc.get_ptr());
        }

        *element_count
            = std::min(requested_element_count, static_cast<int64_t>(_engine_ids.size()));
    }
}

void Engine_heuristic_descriptor::get_heuristic_mode(hipdnnBackendAttributeType_t attribute_type,
                                                     int64_t requested_element_count,
                                                     int64_t* element_count,
                                                     void* array_of_elements) const
{
    THROW_IF_NE(
        attribute_type,
        HIPDNN_TYPE_HEUR_MODE,
        HIPDNN_STATUS_BAD_PARAM,
        "Engine_heuristic_descriptor failed to get heuristic mode: Invalid attribute type.");

    THROW_IF_NE(requested_element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_heuristic_descriptor failed to get heuristic mode: Invalid element count.");

    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine_heuristic_descriptor failed to get heuristic mode: Null pointer.");

    if(element_count != nullptr)
    {
        *element_count = 1;
    }

    auto heur_mode_out = static_cast<hipdnnBackendHeurMode_t*>(array_of_elements);
    *heur_mode_out = _heuristic_mode;
}

std::shared_ptr<const Graph_descriptor> Engine_heuristic_descriptor::get_graph() const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "Engine_heuristic_descriptor::get_graph() failed: Not finalized.");

    return _graph;
}

hipdnnBackendDescriptorType_t Engine_heuristic_descriptor::get_static_type()
{
    return HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
}

} // namespace hipdnn_backend
