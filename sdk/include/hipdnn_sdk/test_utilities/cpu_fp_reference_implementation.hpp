// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_implementation_interface.hpp>
#include <numeric>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
#endif

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

    void batchnorm_bwd(const Tensor& dy,
                       const Tensor& x,
                       const Tensor& mean,
                       const Tensor& invVariance,
                       const Tensor& scale,
                       Tensor& dx,
                       Tensor& dscale,
                       Tensor& dbias) override
    {
        if(x.dims().size() != 4)
        {
            throw std::runtime_error("Batchnorm backward requires a 4D tensor.");
        }

        int64_t n_batches = x.dims().at(0);
        std::vector<int64_t> channels(static_cast<size_t>(x.dims().at(1)));
        std::iota(channels.begin(), channels.end(), 0);
        int64_t height = x.dims().at(2);
        int64_t width = x.dims().at(3);
        int64_t nhw = n_batches * height * width; // Total elements per channel

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto channel_mean = mean.get_host_value<Mean_variance_data_type>(0, cidx, 0, 0);
            auto channel_inv_variance = invVariance.get_host_value<Mean_variance_data_type>(0, cidx, 0, 0);
            auto channel_scale = scale.get_host_value<Scale_bias_data_type>(0, cidx, 0, 0);

            // Calculate dot product of (x - mean) * dy and sum of dy for this channel
            Mean_variance_data_type dot_product = 0;
            Mean_variance_data_type dy_sum = 0;

            for(int row = 0; row < height; row++)
            {
                for(int column = 0; column < width; column++)
                {
                    for(int bidx = 0; bidx < n_batches; bidx++)
                    {
                        auto x_val = static_cast<Mean_variance_data_type>(
                            x.get_host_value<Input_data_type>(bidx, cidx, row, column));
                        auto dy_val = static_cast<Mean_variance_data_type>(
                            dy.get_host_value<Input_data_type>(bidx, cidx, row, column));
                        
                        Mean_variance_data_type x_hat = x_val - channel_mean;
                        dot_product += x_hat * dy_val;
                        dy_sum += dy_val;
                    }
                }
            }

            // Per channel:
            // - dscale = invVariance * ∑ (x_hat * dy)  
            // - dbias = ∑ dy

            dscale.set_host_value<Scale_bias_data_type>(
                0, cidx, 0, 0, 
                static_cast<Scale_bias_data_type>(dot_product * channel_inv_variance));
            
            dbias.set_host_value<Scale_bias_data_type>(
                0, cidx, 0, 0, 
                static_cast<Scale_bias_data_type>(dy_sum));

            // Calculate dx for this channel
            // training: dx = scale * inv_variance * (dy - (dbias / nhw) - (x_hat) * mean((x - mean) * dy))
            Mean_variance_data_type dy_mean = dy_sum / static_cast<Mean_variance_data_type>(nhw);
            Mean_variance_data_type k = dot_product * channel_inv_variance * channel_inv_variance / static_cast<Mean_variance_data_type>(nhw);

            for(int row = 0; row < height; row++)
            {
                for(int column = 0; column < width; column++)
                {
                    for(int bidx = 0; bidx < n_batches; bidx++)
                    {
                        auto x_val = static_cast<Mean_variance_data_type>(
                            x.get_host_value<Input_data_type>(bidx, cidx, row, column));
                        auto dy_val = static_cast<Mean_variance_data_type>(
                            dy.get_host_value<Input_data_type>(bidx, cidx, row, column));
                        
                        Mean_variance_data_type x_hat = x_val - channel_mean;
                        Mean_variance_data_type dx_val = (dy_val - dy_mean - x_hat * k) 
                                                       * channel_inv_variance 
                                                       * static_cast<Mean_variance_data_type>(channel_scale);
                        
                        dx.set_host_value<Input_data_type>(
                            bidx, cidx, row, column, 
                            static_cast<Input_data_type>(dx_val));
                    }
                }
            }
        });

        dx.memory().mark_host_modified();
        dscale.memory().mark_host_modified();
        dbias.memory().mark_host_modified();
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
