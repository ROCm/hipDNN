// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_implementation_interface.hpp>
#include <numeric>
#include <vector>
#include <execution>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T, class U, class V = U>
class Cpu_fp_reference_implementation : public Reference_implementation_interface
{
public:
    Cpu_fp_reference_implementation() = default;
    ~Cpu_fp_reference_implementation() override = default;

    void batchnorm_fwd_inference(const Test_tensor& input,
                 const Test_tensor& scale,
                 const Test_tensor& bias,
                 const Test_tensor& estimatedMean,
                 const Test_tensor& estimatedVariance,
                 Test_tensor& output,
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

        std::for_each(std::execution::par, channels.begin(), channels.end(), [&](int64_t cidx) {
            V mean = get_value<V>(estimatedMean, 0, cidx, 0, 0);
            V variance = get_value<V>(estimatedVariance, 0, cidx, 0, 0);
            double invert_var
                = 1.0 / sqrt(static_cast<double>(variance) + epsilon);
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batches; bidx++)
                    { // via mini_batch
                        double elem_std
                            = static_cast<double>(get_value<T>(input, bidx, cidx, row, column))
                              - static_cast<double>(mean);
                        double inhat = elem_std * invert_var;
                        set_value<T>(
                            output,
                            bidx,
                            cidx,
                            row,
                            column,
                            static_cast<T>(
                                (static_cast<double>(get_value<U>(scale, 0, cidx, 0, 0)) * inhat)
                                + static_cast<double>(get_value<U>(bias, 0, cidx, 0, 0))));
                    }
                }
            }
        });

        output.memory().mark_host_modified(); // Mark output memory as modified on host
    }

private:
    template <typename TUV>
    TUV get_value(
        const Test_tensor& tensor, int64_t bidx, int64_t cidx, int64_t row, int64_t column) const
    {
        int64_t index = get_index(tensor, bidx, cidx, row, column);
        const auto* data = tensor.memory().host_data<TUV>();
        return data[index];
    }

    template <typename TUV>
    void set_value(Test_tensor& tensor,
                   int64_t bidx,
                   int64_t cidx,
                   int64_t row,
                   int64_t column,
                   TUV value) const
    {
        int64_t index = get_index(tensor, bidx, cidx, row, column);
        auto* data = tensor.memory().host_data<TUV>();
        data[index] = value;
    }

    int64_t get_index(
        const Test_tensor& tensor, int64_t bidx, int64_t cidx, int64_t row, int64_t column) const
    {
        const auto& strides = tensor.strides();
        return (bidx * strides[0]) + (cidx * strides[1]) + (row * strides[2])
               + (column * strides[3]);
    }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
