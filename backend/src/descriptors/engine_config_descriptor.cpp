// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_config_descriptor.hpp"
#include "engine_descriptor.hpp"
#include "error.hpp"
#include "graph_descriptor.hpp"
#include "handle/handle.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"
#include <hipdnn_sdk/data_objects/engine_config_generated.h>

namespace hipdnn_backend
{

Engine_config_descriptor::Engine_config_descriptor()
{
    _engine_config_data = std::make_unique<hipdnn_sdk::data_objects::EngineConfigT>();
}

void Engine_config_descriptor::finalize()
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_config_descriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_engine,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine_config_descriptor::finalize() failed: Engine is not set.");

    auto graph = _engine->get_graph();
    auto handle = graph->get_handle();
    auto plugin_resource_manager = handle->get_plugin_resource_manager();

    auto engine_id = _engine->get_engine_id();

    auto engine_config_plugin_data = get_serialized_engine_config();
    auto workspace_size = static_cast<int64_t>(plugin_resource_manager->get_workspace_size(
        engine_id, &engine_config_plugin_data, graph.get()));

    THROW_IF_LT(workspace_size,
                0,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "Engine_config_descriptor::set_max_workspace_size() failed: "
                "Max workspace size cannot be negative.");

    _max_workspace_size = workspace_size;
    hipdnnBackendDescriptorImpl<Engine_config_descriptor>::finalize();
}

void Engine_config_descriptor::get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                             hipdnnBackendAttributeType_t attribute_type,
                                             int64_t requested_element_count,
                                             int64_t* element_count,
                                             void* array_of_elements) const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Engine_config_descriptor::get_attribute() failed: Not finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        get_engine(attribute_type, requested_element_count, element_count, array_of_elements);
        break;
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
        get_max_workspace_size(
            attribute_type, requested_element_count, element_count, array_of_elements);
        break;
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
                                          void* array_of_elements) const
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

    hipdnnBackendDescriptor::pack_descriptor(_engine, array_of_elements);
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

void Engine_config_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
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
        break;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Engine_config_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }

    // reset the serialized buffer when an attribute is set to ensure it's not cached out of date.
    _engine_config_serialized_buffer = flatbuffers::DetachedBuffer();
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

    auto engine = hipdnnBackendDescriptor::unpack_descriptor<const Engine_descriptor>(
        array_of_elements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "Engine_config_descriptor failed to set engine: Engine is null.");

    THROW_IF_FALSE(engine->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Engine_config_descriptor failed to set engine: "
                   "Engine is not finalized.");

    _engine = engine;
    _engine_config_data->engine_id = _engine->get_engine_id();
}

std::shared_ptr<const Engine_descriptor> Engine_config_descriptor::get_engine() const
{
    THROW_IF_FALSE(is_finalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "Engine_config_descriptor::get_engine() failed: Not finalized.");
    return _engine;
}

hipdnnBackendDescriptorType_t Engine_config_descriptor::get_static_type()
{
    return HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
}

hipdnnPluginConstData_t Engine_config_descriptor::get_serialized_engine_config() const
{
    if(_engine_config_serialized_buffer.size() == 0)
    {
        THROW_IF_NULL(_engine,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "Engine_config_descriptor::get_serialized_engine_config: engine is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_sdk::data_objects::EngineConfig::Pack(builder, _engine_config_data.get()));
        _engine_config_serialized_buffer = builder.Release();
    }

    return {.ptr = _engine_config_serialized_buffer.data(),
            .size = _engine_config_serialized_buffer.size()};
}

} // namespace hipdnn_backend
