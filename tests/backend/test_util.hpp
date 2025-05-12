// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

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

} // namespace test_util