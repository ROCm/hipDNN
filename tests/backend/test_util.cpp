// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_util.hpp"
#include "hipdnn_backend.h"

#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <gtest/gtest.h>

namespace test_util
{

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

void create_tensor(hipdnnBackendDescriptor_t* tensor,
                   const int64_t* dims,
                   int64_t dims_count,
                   hipdnnDataType_t data_type)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, tensor),
              HIPDNN_STATUS_SUCCESS);
    populate_tensor(*tensor, dims, dims_count, data_type, true);
}

void populate_tensor(hipdnnBackendDescriptor_t tensor,
                     const int64_t* dims,
                     int64_t dims_count,
                     hipdnnDataType_t data_type,
                     bool finalize)
{
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  tensor, HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, dims_count, dims),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(
                  tensor, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_INT32, 1, &data_type),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(tensor), HIPDNN_STATUS_SUCCESS);
    }
}

void create_input_tensor(hipdnnBackendDescriptor_t* tensor, hipdnnDataType_t data_type)
{
    int64_t dims[] = {1, 3, 224, 224}; // Batch, Channels, Height, Width
    create_tensor(tensor, dims, 4, data_type);
}

void create_output_tensor(hipdnnBackendDescriptor_t* tensor, hipdnnDataType_t data_type)
{
    int64_t dims[] = {1, 1000, 1, 1}; // Batch, Classes
    create_tensor(tensor, dims, 4, data_type);
}

void create_variant_pack(hipdnnBackendDescriptor_t* variant_pack)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, variant_pack),
              HIPDNN_STATUS_SUCCESS);
    populate_variant_pack(*variant_pack);
}

void populate_variant_pack(hipdnnBackendDescriptor_t variant_pack,
                           hipdnnBackendDescriptor_t input_tensor,
                           hipdnnBackendDescriptor_t output_tensor,
                           bool finalize = true)
{
    ASSERT_NE(input_tensor, nullptr) << "Input tensor must not be null";
    ASSERT_NE(output_tensor, nullptr) << "Output tensor must not be null";

    // Set the tensors on the variant pack
    ASSERT_EQ(hipdnnBackendSetAttribute(variant_pack,
                                        HIPDNN_ATTR_VARIANT_PACK_INPUT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &input_tensor),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnBackendSetAttribute(variant_pack,
                                        HIPDNN_ATTR_VARIANT_PACK_OUTPUT,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &output_tensor),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        ASSERT_EQ(hipdnnBackendFinalize(variant_pack), HIPDNN_STATUS_SUCCESS);
    }
}

} // namespace test_util