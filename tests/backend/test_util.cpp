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
                          int64_t gidx)
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
}

void populate_finalized_test_engine(hipdnnBackendDescriptor_t engine,
                                    hipdnnBackendDescriptor_t* graph,
                                    int64_t gidx)
{
    populate_test_engine(engine, graph, gidx);
    ASSERT_EQ(hipdnnBackendFinalize(engine), HIPDNN_STATUS_SUCCESS);
}

void create_test_engine(hipdnnBackendDescriptor_t* engine,
                        hipdnnBackendDescriptor_t* graph,
                        int64_t gidx)
{
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, engine),
              HIPDNN_STATUS_SUCCESS);
    populate_finalized_test_engine(*engine, graph, gidx);
}

void populate_test_engine_config(hipdnnBackendDescriptor_t* engine_config,
                                 hipdnnBackendDescriptor_t* engine,
                                 hipdnnBackendDescriptor_t* graph,
                                 int64_t gidx)
{
    create_test_engine(engine, graph, gidx);
    ASSERT_EQ(hipdnnBackendSetAttribute(*engine_config,
                                        HIPDNN_ATTR_ENGINECFG_ENGINE,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        engine),
              HIPDNN_STATUS_SUCCESS);
}

} // namespace test_util