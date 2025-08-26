// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

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

template <class Input_data_type,
          class Scale_bias_data_type,
          class Mean_variance_data_type = Scale_bias_data_type>
class Reference_implementation_interface
{
public:
    virtual ~Reference_implementation_interface() = default;

    virtual void
        batchnorm_fwd_inference(const Tensor_interface<Input_data_type>& input,
                                const Tensor_interface<Scale_bias_data_type>& scale,
                                const Tensor_interface<Scale_bias_data_type>& bias,
                                const Tensor_interface<Mean_variance_data_type>& estimated_mean,
                                const Tensor_interface<Mean_variance_data_type>& estimated_variance,
                                Tensor_interface<Input_data_type>& output,
                                double epsilon)
        = 0;

    // Could call this bwd_training or bwd_propagation
    virtual void batchnorm_bwd(const Tensor_interface<Input_data_type>& dy,
                               const Tensor_interface<Input_data_type>& x,
                               const Tensor_interface<Mean_variance_data_type>& mean,
                               const Tensor_interface<Mean_variance_data_type>& inv_variance,
                               const Tensor_interface<Scale_bias_data_type>& scale,
                               Tensor_interface<Input_data_type>& dx,
                               Tensor_interface<Scale_bias_data_type>& dscale,
                               Tensor_interface<Scale_bias_data_type>& dbias)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)
