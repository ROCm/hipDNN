// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/tensor.hpp>
#include <map>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

class Reference_implementation_interface
{
public:
    virtual ~Reference_implementation_interface() = default;

    virtual void batchnorm_fwd_inference(const Tensor& input,
                                         const Tensor& scale,
                                         const Tensor& bias,
                                         const Tensor& estimated_mean,
                                         const Tensor& estimated_variance,
                                         Tensor& output,
                                         double epsilon)
        = 0;

    // Could call this bwd_training or bwd_propagation
    virtual void batchnorm_bwd(const Tensor& dy,
                               const Tensor& x,
                               const Tensor& mean,
                               const Tensor& inv_variance,
                               const Tensor& scale,
                               Tensor& dx,
                               Tensor& dscale,
                               Tensor& dbias)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
