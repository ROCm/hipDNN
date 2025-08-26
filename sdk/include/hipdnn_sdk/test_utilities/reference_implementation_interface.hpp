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
        batchnorm_fwd_inference(const ITensor<Input_data_type>& input,
                                const ITensor<Scale_bias_data_type>& scale,
                                const ITensor<Scale_bias_data_type>& bias,
                                const ITensor<Mean_variance_data_type>& estimated_mean,
                                const ITensor<Mean_variance_data_type>& estimated_variance,
                                ITensor<Input_data_type>& output,
                                double epsilon)
        = 0;

    // Could call this bwd_training or bwd_propagation
    virtual void batchnorm_bwd(const ITensor<Input_data_type>& dy,
                               const ITensor<Input_data_type>& x,
                               const ITensor<Mean_variance_data_type>& mean,
                               const ITensor<Mean_variance_data_type>& inv_variance,
                               const ITensor<Scale_bias_data_type>& scale,
                               ITensor<Input_data_type>& dx,
                               ITensor<Scale_bias_data_type>& dscale,
                               ITensor<Scale_bias_data_type>& dbias)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)
