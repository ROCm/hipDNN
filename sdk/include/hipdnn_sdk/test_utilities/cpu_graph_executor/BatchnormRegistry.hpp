// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureKey.hpp>
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

// template <typename InputT, typename ScaleBiasT, typename MeanVarianceT>
// struct BatchnormSignature
// {
//     using InputDataType = InputT;
//     using ScaleBiasDataType = ScaleBiasT;
//     using MeanVarianceDataType = MeanVarianceT;
// };

//THIS DOESNT WORK AT ALL, need template params
// constexpr std::array<BatchnormSignatureRegistryKey<hipdnn_sdk::data_objects::DataType::FLOAT,
//                                                    hipdnn_sdk::data_objects::DataType::FLOAT,
//                                                    hipdnn_sdk::data_objects::DataType::FLOAT>,
//                      1>
//     allBatchnormSignatures = {
//         BatchnormSignatureRegistryKey<hipdnn_sdk::data_objects::DataType::FLOAT,
//                                       hipdnn_sdk::data_objects::DataType::FLOAT,
//                                       hipdnn_sdk::data_objects::DataType::FLOAT>{},
//         //BatchnormSignatureRegistryKey<hipdnn_sdk::data_objects::DataType::HALF,
//         //                              hipdnn_sdk::data_objects::DataType::HALF,
//         //                              hipdnn_sdk::data_objects::DataType::HALF>{}
// };

// Registry key: compile-time template

constexpr std::array allBatchnormSignatures
    = {BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF)};

//These functions are esentially looping over std::array and creating all the executors.
template <std::size_t... Is>
void registerBatchnormExecutors(std::index_sequence<Is...>)
{
    ((batchnormRegistry()[allBatchnormSignatures[Is].toSignatureKey()]
      = std::make_unique<BatchnormExecutor<allBatchnormSignatures[Is]>>()),
     ...);
}

inline void initializeBatchnormRegistry()
{
    registerBatchnormExecutors(std::make_index_sequence<allBatchnormSignatures.size()>{});
}

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        initializeBatchnormRegistry();
        // constexpr auto something
        //     = BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::FLOAT,
        //                                     hipdnn_sdk::data_objects::DataType::FLOAT,
        //                                     hipdnn_sdk::data_objects::DataType::FLOAT);

        // auto stuff = BatchnormExecutor<something>();
        // std::ignore = stuff;

        // for(constexpr auto whatever : allBatchnormSignatures)
        // {
        //     auto stuff = BatchnormExecutor<whatever>();
        //     // std::ignore = stuff;
        // }

        // std::array<std::unique_ptr<IGenericBatchnormExecutor>, 1> executors{
        //     std::make_unique<BatchnormExecutor<
        //         BatchnormSignatureRegistryKey<hipdnn_sdk::data_objects::DataType::FLOAT,
        //                                       hipdnn_sdk::data_objects::DataType::FLOAT,
        //                                       hipdnn_sdk::data_objects::DataType::FLOAT>{}>>(),
        //     //std::make_unique<BatchnormExecutor<BatchnormSignatureRegistryKey<half, half, half>>>()
        // };

        // for(auto& executor : executors)
        // {
        //     batchnormRegistry()[executor->signatureKey()] = std::move(executor);
        // }
    }
};

inline BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
}
