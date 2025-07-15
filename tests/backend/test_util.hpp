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

void create_and_populate_batchnorm_node(Graph& graph);

void* allocate_tensor_memory([[maybe_unused]] const int64_t* dims,
                             [[maybe_unused]] size_t dims_count,
                             [[maybe_unused]] hipdnnBackendAttributeType_t data_type,
                             [[maybe_unused]] bool initialize);

void free_tensor_memory(void* data_ptr);

void populate_variant_pack_with_mappings(
    hipdnnBackendDescriptor_t variant_pack,
    const std::unordered_map<int64_t, void*>& data_ptr_mappings,
    void* workspace = nullptr);

void create_and_initialize_backend_descriptor(hipdnnBackendDescriptor_t* backend_descriptor,
                                              const flatbuffers::DetachedBuffer& serialized_graph);

void extract_tensor_info_from_graph(
    const flatbuffers::DetachedBuffer& serialized_graph,
    std::unordered_map<int64_t, std::string>& uid_to_name_map,
    std::unordered_map<std::string, int64_t>& name_to_uid_map,
    std::unordered_map<int64_t, std::vector<int64_t>>& uid_to_dims_map);
} // namespace test_util