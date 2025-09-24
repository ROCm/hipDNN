// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>

namespace hipdnn_sdk::test_utilities
{

using SignatureKey = std::variant<BatchnormSignatureRegistryKey>;

inline std::unordered_map<SignatureKey, std::unique_ptr<IGraphNodePlanBuilder>>&
    planBuilderRegistry()
{
    static std::unordered_map<SignatureKey, std::unique_ptr<IGraphNodePlanBuilder>> registry;
    return registry;
}

//TODO FIX DOCUMENTATION FOR THIS.
template <std::size_t... Is>
void registerBatchnormFwdInferencePlanBuilders(std::index_sequence<Is...>)
{
    ((planBuilderRegistry()[ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is]] = std::make_unique<
          BatchnormFwdInferencePlanBuilder<ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is]>>()),
     ...);
}

inline void initializeBatchnormRegistry()
{
    registerBatchnormFwdInferencePlanBuilders(
        std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_SIGNATURES.size()>{});
}

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        initializeBatchnormRegistry();
    }
};

inline BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
