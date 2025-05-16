// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_util.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/logger.hpp>

#include <gtest/gtest.h>

namespace test_util
{

void create_test_handle(hipdnnHandle_t* handle)
{
    ASSERT_EQ(hipdnnCreate(handle), HIPDNN_STATUS_SUCCESS);
}

void create_test_graph(hipdnnBackendDescriptor_t* descriptor)
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
}

void populate_test_engine(hipdnnBackendDescriptor_t engine,
                          hipdnnBackendDescriptor_t* graph,
                          int64_t gidx,
                          bool finalize)
{
    create_test_graph(graph);
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
                        int64_t gidx)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, engine),
              HIPDNN_STATUS_SUCCESS);
    populate_test_engine(*engine, graph, gidx, true);
}

void populate_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                                 hipdnnBackendDescriptor_t* engine,
                                 hipdnnBackendDescriptor_t* graph,
                                 int64_t gidx,
                                 bool finalize)
{
    create_test_engine(engine, graph, gidx);
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
                               int64_t gidx,
                               bool finalize)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, engine_config),
              HIPDNN_STATUS_SUCCESS);
    populate_test_engine_config(engine_config, engine, graph, gidx, finalize);
}

void populate_test_execution_plan(hipdnnBackendDescriptor_t* execution_plan,
                                  hipdnnHandle_t* handle,
                                  hipdnnBackendDescriptor_t* engine_config,
                                  hipdnnBackendDescriptor_t* engine,
                                  hipdnnBackendDescriptor_t* graph,
                                  int64_t gidx,
                                  bool finalize)
{
    create_test_handle(handle);
    ASSERT_EQ(
        hipdnnBackendSetAttribute(
            *execution_plan, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, handle),
        HIPDNN_STATUS_SUCCESS);

    create_test_engine_config(engine_config, engine, graph, gidx, true);
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
                             [[maybe_unused]] int64_t dims_count,
                             [[maybe_unused]] DataType_t data_type,
                             [[maybe_unused]] bool initialize)
{
    // TODO: Implement memory allocation logic based on the data type and dimensions
    // For now, just return a dummy pointer
    void* memory = malloc(0);
    return memory;
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

void create_batchnorm_graph(const std::shared_ptr<Tensor_attributes>& x,
                            const std::shared_ptr<Tensor_attributes>& scale,
                            const std::shared_ptr<Tensor_attributes>& bias,
                            const std::vector<int64_t>& input_dims,
                            const std::vector<int64_t>& input_strides,
                            DataType_t data_type,
                            const std::string& node_name)
{
    x->set_dim(input_dims)
        .set_stride(input_strides)
        .set_data_type(data_type)
        .set_name(node_name + "::Input");

    scale->set_data_type(data_type).set_name(node_name + "::Scale");

    bias->set_data_type(data_type).set_name(node_name + "::Bias");
}

void create_and_initialize_backend_descriptor(hipdnnBackendDescriptor_t backend_descriptor,
                                              const flatbuffers::DetachedBuffer& serialized_graph)
{
    ASSERT_NE(backend_descriptor, nullptr);

    auto status = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                                &backend_descriptor);

    ASSERT_NE(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &backend_descriptor, serialized_graph.data(), serialized_graph.size());
    ASSERT_NE(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(backend_descriptor);
    ASSERT_NE(status, HIPDNN_STATUS_SUCCESS);
}

void create_and_populate_batchnorm_node(Graph& graph)
{
    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("Input")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 64, 28, 28}); // NCHW format

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType_t::FLOAT).set_dim({1, 64, 1, 1});

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType_t::FLOAT).set_dim({1, 64, 1, 1});

    auto mean = std::make_shared<Tensor_attributes>();
    mean->set_uid(4).set_name("Mean").set_data_type(DataType_t::FLOAT).set_dim({1, 64, 1, 1});

    auto inv_variance = std::make_shared<Tensor_attributes>();
    inv_variance->set_uid(5)
        .set_name("InvVariance")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 64, 1, 1});

    // Create BatchnormInference node
    Batchnorm_inference_attributes attrs;

    // TODO: add epsilon

    [[maybe_unused]] auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attrs);

    // Build the operation graph
    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();
}

void extract_tensor_info_from_graph(const flatbuffers::DetachedBuffer& serialized_graph)
{
    // Currently using FlatBuffers to directly deserialize the graph and extract tensor information.
    // In the future, we will retrieve this information through the backend descriptor's get_attribute API.
    //auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(serialized_graph.data());
    //ASSERT_NE(deserialized_graph, nullptr);
    std::unordered_map<int64_t, std::string> uid_to_name_map;
    std::unordered_map<std::string, int64_t> name_to_uid_map;
    std::unordered_map<int64_t, std::vector<int64_t>> uid_to_dims_map;
    // Clear output maps
    uid_to_name_map.clear();
    name_to_uid_map.clear();
    uid_to_dims_map.clear();

    // Deserialize the graph using FlatBuffers
    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr) << "Failed to deserialize graph";

    // Extract all tensor information from the deserialized graph
    for(const auto& tensor : deserialized_graph->tensors)
    {
        int64_t uid = tensor->uid;
        std::string name = tensor->name;

        // Store mappings
        uid_to_name_map[uid] = name;
        name_to_uid_map[name] = uid;

        // Store dimensions if available
        if(!tensor->dims.empty())
        {
            uid_to_dims_map[uid] = tensor->dims;
        }
    }

    ASSERT_FALSE(uid_to_name_map.empty());
}

} // namespace test_util