// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GenericBatchnormExecutor.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

constexpr std::array ALL_BATCHNORM_SIGNATURES
    = {BatchnormSignatureRegistryKey{
           .INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType::FLOAT,
           .SCALE_BIAS_DATA_TYPE = hipdnn_sdk::data_objects::DataType::FLOAT,
           .MEAN_VARIANCE_DATA_TYPE = hipdnn_sdk::data_objects::DataType::FLOAT},
       BatchnormSignatureRegistryKey{
           .INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType::HALF,
           .SCALE_BIAS_DATA_TYPE = hipdnn_sdk::data_objects::DataType::HALF,
           .MEAN_VARIANCE_DATA_TYPE = hipdnn_sdk::data_objects::DataType::HALF}};

inline std::unordered_map<BatchnormSignatureRegistryKey,
                          std::unique_ptr<IGenericBatchnormExecutor>>&
    batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureRegistryKey,
                              std::unique_ptr<IGenericBatchnormExecutor>>
        registry;
    return registry; //
}

/**
 * @brief Registers all batchnorm executors at compile-time using fold expressions
 * 
 * This variadic template function expands a parameter pack of indices to register
 * multiple BatchnormExecutor instances in the batchnormRegistry. Each executor
 * is instantiated with its corresponding signature from ALL_BATCHNORM_SIGNATURES array.
 * 
 * @tparam Is Variadic template parameter pack of indices
 * @param std::index_sequence<Is...> Compile-time sequence of indices used to
 *        iterate through ALL_BATCHNORM_SIGNATURES array
 * 
 * @note Uses C++17 fold expression with comma operator to expand and execute
 *       the registration for each index in the sequence
 */
template <std::size_t... Is>
void registerBatchnormExecutors(std::index_sequence<Is...>)
{
    ((batchnormRegistry()[ALL_BATCHNORM_SIGNATURES[Is]]
      = std::make_unique<BatchnormExecutor<ALL_BATCHNORM_SIGNATURES[Is]>>()),
     ...);
}

inline void initializeBatchnormRegistry()
{
    registerBatchnormExecutors(std::make_index_sequence<ALL_BATCHNORM_SIGNATURES.size()>{});
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
}
