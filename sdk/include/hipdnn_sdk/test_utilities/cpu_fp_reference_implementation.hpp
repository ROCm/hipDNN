// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_implementation_interface.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class Input_data_type,
          class Scale_bias_data_type,
          class Mean_variance_data_type = Scale_bias_data_type>
class Cpu_fp_reference_implementation : public Reference_implementation_interface
{
public:
    Cpu_fp_reference_implementation() = default;
    ~Cpu_fp_reference_implementation() override = default;

    void batchnorm_fwd_inference(const Tensor& input,
                                 const Tensor& scale,
                                 const Tensor& bias,
                                 const Tensor& estimatedMean,
                                 const Tensor& estimatedVariance,
                                 Tensor& output,
                                 double epsilon) override
    {
        if(input.dims().size() != 4)
        {
            throw std::runtime_error("Batchnorm inference requires a 4D tensor.");
        }

        int64_t n_batches = input.dims().at(0);
        std::vector<int64_t> channels(static_cast<size_t>(input.dims().at(1)));
        std::iota(channels.begin(), channels.end(), 0);
        int64_t height = input.dims().at(2);
        int64_t width = input.dims().at(3);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto mean = estimatedMean.get_host_value<Mean_variance_data_type>(0, cidx, 0, 0);
            auto variance
                = estimatedVariance.get_host_value<Mean_variance_data_type>(0, cidx, 0, 0);
            Mean_variance_data_type invert_var
                = static_cast<Mean_variance_data_type>(1.0f)
                  / sqrt_internal(variance + static_cast<Mean_variance_data_type>(epsilon));
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batches; bidx++)
                    { // via mini_batch
                        auto in = static_cast<Mean_variance_data_type>(
                            input.get_host_value<Input_data_type>(bidx, cidx, row, column));
                        Mean_variance_data_type elem_std = in - mean;
                        Mean_variance_data_type inhat = elem_std * invert_var;
                        output.set_host_value<Input_data_type>(
                            bidx,
                            cidx,
                            row,
                            column,
                            static_cast<Input_data_type>(
                                (scale.get_host_value<Scale_bias_data_type>(0, cidx, 0, 0)
                                 * static_cast<Scale_bias_data_type>(inhat))
                                + bias.get_host_value<Scale_bias_data_type>(0, cidx, 0, 0)));
                    }
                }
            }
        });

        output.memory().mark_host_modified(); // Mark output memory as modified on host
    }

private:
    double sqrt_internal(double value) const
    {
        return std::sqrt(value);
    }

    float sqrt_internal(float value) const
    {
        return std::sqrtf(value);
    }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
