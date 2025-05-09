// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"

namespace test_util
{

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
void create_tensor(hipdnnBackendDescriptor_t* tensor,
                   const int64_t* dims,
                   int64_t dims_count,
                   hipdnnDataType_t data_type = HIPDNN_DATA_FLOAT);

void create_tensor(hipdnnBackendDescriptor_t* tensor,
                   const int64_t* dims,
                   int64_t dims_count,
                   hipdnnDataType_t data_type = HIPDNN_DATA_FLOAT);

void populate_tensor(hipdnnBackendDescriptor_t tensor,
                     const int64_t* dims,
                     int64_t dims_count,
                     hipdnnDataType_t data_type = HIPDNN_DATA_FLOAT,
                     bool finalize = true);

void create_input_tensor(hipdnnBackendDescriptor_t* tensor,
                         hipdnnDataType_t data_type = HIPDNN_DATA_FLOAT);

void create_output_tensor(hipdnnBackendDescriptor_t* tensor,
                          hipdnnDataType_t data_type = HIPDNN_DATA_FLOAT);

void create_variant_pack(hipdnnBackendDescriptor_t* variant_pack);

void populate_variant_pack(hipdnnBackendDescriptor_t variant_pack, bool finalize = true);

} // namespace test_util