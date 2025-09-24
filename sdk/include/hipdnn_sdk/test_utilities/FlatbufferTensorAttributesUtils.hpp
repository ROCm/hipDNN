// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_sdk::test_utilities
{

hipdnn_sdk::data_objects::TensorAttributesT
    unpackTensorAttributes(const hipdnn_sdk::data_objects::TensorAttributes& tensorAttributes)
{
    hipdnn_sdk::data_objects::TensorAttributesT tensorAttributesT;
    tensorAttributes.UnPackTo(&tensorAttributesT);

    return tensorAttributesT;
}

}
