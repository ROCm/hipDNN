// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBuilder.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

// 1. Define the signature POD
struct FwdBatchnormSignatureFloat
{
    static constexpr auto INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType_FLOAT;
    static constexpr auto NODE_ATTRIBUTES_TYPE
        = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes;
};
static_assert(BatchnormSignatureDescriptor<FwdBatchnormSignatureFloat>);

struct FwdBatchnormSignatureHalf
{
    static constexpr auto INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType_HALF;
    static constexpr auto NODE_ATTRIBUTES_TYPE
        = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes;
};
static_assert(BatchnormSignatureDescriptor<FwdBatchnormSignatureHalf>);

}
}
