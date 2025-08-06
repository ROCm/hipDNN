// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_util.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

#include <gtest/gtest.h>

namespace test_util
{

void create_test_handle(hipdnnHandle_t* handle)
{
    ASSERT_EQ(hipdnnCreate(handle), HIPDNN_STATUS_SUCCESS);
}

void create_test_graph(hipdnnBackendDescriptor_t* descriptor, hipdnnHandle_t handle)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      &tensor_attributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serialized_graph = builder.Release();

    ASSERT_EQ(hipdnnBackendCreateAndDeserializeGraph_ext(
                  descriptor, serialized_graph.data(), serialized_graph.size()),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(
                  *descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
              HIPDNN_STATUS_SUCCESS);
}

void populate_test_engine(hipdnnBackendDescriptor_t engine,
                          hipdnnBackendDescriptor_t* graph,
                          hipdnnHandle_t handle,
                          int64_t gidx,
                          bool finalize)
{
    if(*graph == nullptr)
    {
        create_test_graph(graph, handle);
    }

    ASSERT_EQ(hipdnnBackendFinalize(*graph), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            engine, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, graph),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &gidx),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(engine), HIPDNN_STATUS_SUCCESS);
    }
}

void create_test_engine(hipdnnBackendDescriptor_t* engine,
                        hipdnnBackendDescriptor_t* graph,
                        hipdnnHandle_t handle,
                        int64_t gidx)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, engine),
              HIPDNN_STATUS_SUCCESS);
    populate_test_engine(*engine, graph, handle, gidx, true);
}

void populate_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                                 hipdnnBackendDescriptor_t* engine,
                                 hipdnnBackendDescriptor_t* graph,
                                 hipdnnHandle_t handle,
                                 int64_t gidx,
                                 bool finalize)
{
    if(*engine == nullptr)
    {
        create_test_engine(engine, graph, handle, gidx);
    }

    ASSERT_EQ(hipdnnBackendSetAttribute(*engine_config,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        engine),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(*engine_config), HIPDNN_STATUS_SUCCESS);
    }
}

void create_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                               hipdnnBackendDescriptor_t* engine,
                               hipdnnBackendDescriptor_t* graph,
                               hipdnnHandle_t handle,
                               int64_t gidx,
                               bool finalize)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, engine_config),
              HIPDNN_STATUS_SUCCESS);
    populate_test_engine_config(engine_config, engine, graph, handle, gidx, finalize);
}

void populate_test_execution_plan(hipdnnBackendDescriptor_t* execution_plan,
                                  hipdnnBackendDescriptor_t* engine_config,
                                  hipdnnBackendDescriptor_t* engine,
                                  hipdnnBackendDescriptor_t* graph,
                                  hipdnnHandle_t handle,
                                  int64_t gidx,
                                  bool finalize)
{
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *execution_plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_SUCCESS);

    if(*engine_config == nullptr)
    {
        create_test_engine_config(engine_config, engine, graph, handle, gidx, true);
    }

    ASSERT_EQ(hipdnnBackendSetAttribute(*execution_plan,
                                        HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        engine_config),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(*execution_plan), HIPDNN_STATUS_SUCCESS);
    }
}

void* allocate_tensor_memory([[maybe_unused]] const int64_t* dims,
                             [[maybe_unused]] size_t dims_count,
                             [[maybe_unused]] hipdnnBackendAttributeType_t data_type,
                             [[maybe_unused]] bool initialize)
{
    // TODO: Implement memory allocation logic based on the data type and dimensions
    // For now, just return a dummy pointer
    void* memory = malloc(0);
    return memory;
}

void free_tensor_memory(void* data_ptr)
{
    if(data_ptr != nullptr)
    {
        free(data_ptr);
    }
}

void set_tensor_mappings_in_variant_pack(hipdnnBackendDescriptor_t variant_pack,
                                         const std::vector<int64_t>& tensor_ids,
                                         const std::vector<void*>& data_ptrs)
{
    ASSERT_EQ(hipdnnBackendSetAttribute(variant_pack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(tensor_ids.size()),
                                        tensor_ids.data()),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(variant_pack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        static_cast<int64_t>(data_ptrs.size()),
                                        data_ptrs.data()),
              HIPDNN_STATUS_SUCCESS);
}

void set_workspace_in_variant_pack(hipdnnBackendDescriptor_t variant_pack, void* workspace)
{
    if(workspace != nullptr)
    {
        ASSERT_EQ(hipdnnBackendSetAttribute(variant_pack,
                                            HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                            HIPDNN_TYPE_VOID_PTR,
                                            1,
                                            &workspace),
                  HIPDNN_STATUS_SUCCESS);
    }
}

void finalize_variant_pack(hipdnnBackendDescriptor_t variant_pack)
{
    ASSERT_EQ(hipdnnBackendFinalize(variant_pack), HIPDNN_STATUS_SUCCESS);
}

void extract_tensor_mappings(const std::unordered_map<int64_t, void*>& data_ptr_mappings,
                             std::vector<int64_t>& tensor_ids,
                             std::vector<void*>& data_ptrs)
{
    for(const auto& [id, data_ptr] : data_ptr_mappings)
    {
        ASSERT_NE(data_ptr, nullptr);
        tensor_ids.push_back(id);
        data_ptrs.push_back(data_ptr);
    }
    ASSERT_EQ(tensor_ids.size(), data_ptrs.size());
    ASSERT_FALSE(tensor_ids.empty());
    ASSERT_FALSE(data_ptrs.empty());
}

void populate_variant_pack_with_mappings(
    hipdnnBackendDescriptor_t variant_pack,
    const std::unordered_map<int64_t, void*>& data_ptr_mappings,
    void* workspace)
{
    std::vector<int64_t> tensor_ids;
    std::vector<void*> data_ptrs;

    extract_tensor_mappings(data_ptr_mappings, tensor_ids, data_ptrs);
    set_tensor_mappings_in_variant_pack(variant_pack, tensor_ids, data_ptrs);
    set_workspace_in_variant_pack(variant_pack, workspace);
    finalize_variant_pack(variant_pack);
}

DataType_t convert_backend_attribute_to_data_type(hipdnnBackendAttributeType_t backend_type)
{
    switch(backend_type)
    {
    case HIPDNN_TYPE_FLOAT:
        return DataType_t::FLOAT;
    default:
        HIPDNN_LOG_WARN("Unsupported backend attribute type");
        return DataType_t::FLOAT;
    }
}

void create_and_initialize_backend_descriptor(hipdnnBackendDescriptor_t* backend_descriptor,
                                              const flatbuffers::DetachedBuffer& serialized_graph,
                                              hipdnnHandle_t handle)
{
    ASSERT_EQ(*backend_descriptor, nullptr);

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        backend_descriptor, serialized_graph.data(), serialized_graph.size());
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *backend_descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle),
        HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(*backend_descriptor);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

flatbuffers::FlatBufferBuilder create_and_populate_batchnorm_node()
{
    return flatbuffer_test_utils::create_valid_batchnorm_graph();
}

void extract_tensor_info_from_graph(
    const flatbuffers::DetachedBuffer& serialized_graph,
    std::unordered_map<int64_t, std::string>& uid_to_name_map,
    std::unordered_map<std::string, int64_t>& name_to_uid_map,
    std::unordered_map<int64_t, std::vector<int64_t>>& uid_to_dims_map)
{
    uid_to_name_map.clear();
    name_to_uid_map.clear();
    uid_to_dims_map.clear();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    // Extract all tensor information from the deserialized graph
    for(const auto& tensor : deserialized_graph->tensors)
    {
        int64_t uid = tensor->uid;
        std::string name = tensor->name;

        uid_to_name_map[uid] = name;
        name_to_uid_map[name] = uid;

        if(!tensor->dims.empty())
        {
            uid_to_dims_map[uid] = tensor->dims;
        }
    }

    ASSERT_FALSE(uid_to_name_map.empty());
}

} // namespace test_util
