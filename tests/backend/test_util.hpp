// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
namespace test_util
{

void create_test_handle(hipdnnHandle_t* handle);

void create_test_graph(hipdnnBackendDescriptor_t* descriptor);

void populate_test_engine(hipdnnBackendDescriptor_t engine,
                          hipdnnBackendDescriptor_t* graph,
                          int64_t gidx,
                          bool finalize = false);

void create_test_engine(hipdnnBackendDescriptor_t* engine,
                        hipdnnBackendDescriptor_t* graph,
                        int64_t gidx);

void populate_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                                 hipdnnBackendDescriptor_t* engine,
                                 hipdnnBackendDescriptor_t* graph,
                                 int64_t gidx,
                                 bool finalize = false);

void create_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                               hipdnnBackendDescriptor_t* engine,
                               hipdnnBackendDescriptor_t* graph,
                               int64_t gidx,
                               bool finalize = false);

void populate_test_execution_plan(hipdnnBackendDescriptor_t* execution_plan,
                                  hipdnnHandle_t* handle,
                                  hipdnnBackendDescriptor_t* engine_config,
                                  hipdnnBackendDescriptor_t* engine,
                                  hipdnnBackendDescriptor_t* graph,
                                  int64_t gidx,
                                  bool finalize = false);

void create_variant_pack(hipdnnBackendDescriptor_t* variant_pack);

void create_and_populate_batchnorm_node(Graph& graph);

void* allocate_tensor_memory([[maybe_unused]] const int64_t* dims,
                             [[maybe_unused]] size_t dims_count,
                             [[maybe_unused]] hipdnnBackendAttributeType_t data_type,
                             [[maybe_unused]] bool initialize);

void create_batchnorm_input_tensors(const std::vector<int64_t>& input_dims,
                                    const std::vector<int64_t>& input_strides,
                                    DataType_t data_type,
                                    const std::string& node_name);

void create_batchnorm_graph(hipdnnBackendDescriptor_t* graph_descriptor,
                            Graph& graph,
                            const std::shared_ptr<Tensor_attributes>& x,
                            const std::shared_ptr<Tensor_attributes>& scale,
                            const std::shared_ptr<Tensor_attributes>& bias,
                            const std::vector<int64_t>& input_dims,
                            const std::vector<int64_t>& input_strides,
                            DataType_t data_type,
                            const std::string& node_name);

void extract_tensor_mappings(const std::unordered_map<int64_t, void*>& data_ptr_mappings,
                             std::vector<int64_t>& tensor_ids,
                             std::vector<void*>& data_ptrs);
void populate_variant_pack_with_mappings(
    hipdnnBackendDescriptor_t variant_pack,
    const std::unordered_map<int64_t, void*>& data_ptr_mappings,
    void* workspace = nullptr);

void create_and_initialize_backend_descriptor(hipdnnBackendDescriptor_t backend_descriptor,
                                              const flatbuffers::DetachedBuffer& serialized_graph);

void extract_tensor_info_from_graph(
    const flatbuffers::DetachedBuffer& serialized_graph,
    std::unordered_map<int64_t, std::string>& uid_to_name_map,
    std::unordered_map<std::string, int64_t>& name_to_uid_map,
    std::unordered_map<int64_t, std::vector<int64_t>>& uid_to_dims_map);
} // namespace test_util