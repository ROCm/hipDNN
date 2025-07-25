// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/test_tensor.hpp>
#include <map>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

template <class T, class U, class V = U>
class Reference_implementation_interface
{
public:
    virtual ~Reference_implementation_interface() = default;

    virtual void execute(const Test_tensor& input,
                         const Test_tensor& scale,
                         const Test_tensor& bias,
                         const Test_tensor& estimatedMean,
                         const Test_tensor& estimatedVariance,
                         Test_tensor& output,
                         V epsilon)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk