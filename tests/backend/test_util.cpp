// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_util.hpp"
#include "hipdnn_backend.h"

#include <hipdnn_sdk/data_objects/graph_generated.h>

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

} // namespace test_util