// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_implementation_interface.hpp>
#include <vector>
#include <numeric>

namespace hipdnn_sdk {
namespace reference_test_utilities {

using namespace hipdnn_sdk::utilities;

template<class T, class V = T>
class Cpu_fp_reference_implementation : public Reference_implementation_interface<T> {
public:

    Cpu_fp_reference_implementation() = default;
    ~Cpu_fp_reference_implementation() override = default;
    
    void execute(const std::map<int64_t, Migratable_memory<T>>& buffers, const TensorAttributesT& tensor_attributes, const BatchnormInferenceAttributesT& batchnorm_attributes) override
    {
        if (tensor_attributes.dims.size() != 4)
        {
            throw std::runtime_error("Batchnorm inference requires a 4D tensor.");
        }

        int64_t n_batches = tensor_attributes.dims[0];
        std::vector<int64_t> channels(static_cast<size_t>(tensor_attributes.dims[1]));
        std::iota(channels.begin(), channels.end(), 0);
        int64_t height    = tensor_attributes.dims[2];
        int64_t width     = tensor_attributes.dims[3];

        (void)n_batches; // Suppress unused variable warning
        (void)height;    // Suppress unused variable warning
        (void)width;     // Suppress unused variable warning

        // Migratable_memory<T> input = buffers.at(batchnorm_attributes.input_id);
        // Migratable_memory<T> output = buffers.at(batchnorm_attributes.output_id);

        const Migratable_memory<T>& scale  = buffers.at(batchnorm_attributes.scale);
        const Migratable_memory<T>& bias   = buffers.at(batchnorm_attributes.bias);
        const Migratable_memory<V>& estimatedMean = buffers.at(batchnorm_attributes.mean.value());
        const Migratable_memory<V>& estimatedVariance = buffers.at(batchnorm_attributes.inv_variance.value());

        (void)scale;  // Suppress unused variable warning
        (void)bias;   // Suppress unused variable warning
        (void)estimatedMean;  // Suppress unused variable warning
        (void)estimatedVariance;  // Suppress unused variable warning

        std::for_each(channels.begin(), channels.end(), [&](int cidx) {
            (void) cidx; // Suppress unused variable warning

        //     V mean           = estimatedMean(0, cidx, 0, 0);
        //     V variance       = estimatedVariance(0, cidx, 0, 0);
        //     double invertVar = 1.0 / sqrt(variance + epsilon);
        //     // process the batch per channel
        //     for(int row = 0; row < height; row++)
        //     { // via rows
        //         for(int column = 0; column < width; column++)
        //         { // via columns
        //             for(int bidx = 0; bidx < n_batches; bidx++)
        //             { // via mini_batch
        //                 double elemStd = static_cast<double>(input(bidx, cidx, row, column)) - mean;
        //                 double inhat   = elemStd * invertVar;
        //                 output(bidx, cidx, row, column) =
        //                     static_cast<T>(scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
        //                 // printf("output: %f\n",scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
        //             }
        //         }
        //     }
        });

    }

private:

    // template <class T, class Tref, class U, class V = U>
    // void batchNormSpatialHostInference(const tensor<T>& input,
    //                                 tensor<Tref>& output,
    //                                 const tensor<U>& scale,
    //                                 const tensor<U>& bias,
    //                                 double epsilon,
    //                                 const tensor<V>& estimatedMean,
    //                                 const tensor<V>& estimatedVariance)
    // {

    //     int n_batches, channels, height, width;
    //     std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    //     par_for(channels, 1, [&](int cidx) { // via channel
    //         V mean           = estimatedMean(0, cidx, 0, 0);
    //         V variance       = estimatedVariance(0, cidx, 0, 0);
    //         double invertVar = 1.0 / sqrt(variance + epsilon);
    //         // process the batch per channel
    //         for(int row = 0; row < height; row++)
    //         { // via rows
    //             for(int column = 0; column < width; column++)
    //             { // via columns
    //                 for(int bidx = 0; bidx < n_batches; bidx++)
    //                 { // via mini_batch
    //                     double elemStd = static_cast<double>(input(bidx, cidx, row, column)) - mean;
    //                     double inhat   = elemStd * invertVar;
    //                     output(bidx, cidx, row, column) =
    //                         static_cast<T>(scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
    //                     // printf("output: %f\n",scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
    //                 }
    //             }
    //         }
    //     });
    // }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
