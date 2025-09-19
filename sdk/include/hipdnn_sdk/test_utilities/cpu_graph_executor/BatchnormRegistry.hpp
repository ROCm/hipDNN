// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GenericBatchnormExecutor.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

inline std::unordered_map<BatchnormSignatureKey, std::unique_ptr<IGenericBatchnormExecutor>>&
    batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureKey, std::unique_ptr<IGenericBatchnormExecutor>>
        _reg;
    return _reg;
}

template <typename InputT, typename ScaleBiasT, typename MeanVarianceT>
struct BatchnormSignature
{
    using InputDataType = InputT;
    using ScaleBiasDataType = ScaleBiasT;
    using MeanVarianceDataType = MeanVarianceT;
};

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        std::array<std::unique_ptr<IGenericBatchnormExecutor>, 2> executors{
            std::make_unique<BatchnormExecutor<BatchnormSignature<float, float, float>>>(),
            std::make_unique<BatchnormExecutor<BatchnormSignature<half, half, half>>>()};

        for(auto& executor : executors)
        {
            batchnormRegistry()[executor->signatureKey()] = std::move(executor);
        }
    }
};

inline BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
}
