// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignature.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/TensorVariant.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

// 2. Define the key for the registry
struct BatchnormSignatureKey
{
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;

    hipdnn_sdk::data_objects::NodeAttributes nodeAttributesType;

    bool operator==(const BatchnormSignatureKey& other) const
    {
        return inputDataType == other.inputDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && nodeAttributesType == other.nodeAttributesType;
    }
};

}
}

//todo, figure out better way to do this. hash cant be inside the hipdnn_sdk namespace
namespace std
{
template <>
struct hash<hipdnn_sdk::test_utilities::BatchnormSignatureKey>
{
    std::size_t operator()(const hipdnn_sdk::test_utilities::BatchnormSignatureKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType))
               ^ (std::hash<int>()(static_cast<int>(k.scaleBiasDataType)) << 1)
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 1)
               ^ (std::hash<int>()(static_cast<int>(k.nodeAttributesType)) << 1);
    }
};
}

namespace hipdnn_sdk
{
namespace test_utilities
{

using BatchnormFn
    = std::function<void(std::any&, std::any&, std::any&, std::any&, std::any&, std::any&, double)>;

// Registry keyed by BatchnormSignatureKey
std::unordered_map<BatchnormSignatureKey, BatchnormFn>& batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureKey, BatchnormFn> _reg;
    return _reg;
}

// Compile-time iteration over variant alternatives
template <typename Variant, typename F, std::size_t... Is>
inline void _forEachVariantAltImpl(F&& f, std::index_sequence<Is...>)
{
    (f(std::type_identity<std::variant_alternative_t<Is, Variant>>{}), ...);
}

template <typename F>
inline void forEachBatchnormSignatureAlt(F&& f)
{
    using Variant = BatchnormSignatureVariants;
    _forEachVariantAltImpl<Variant>(std::forward<F>(f),
                                    std::make_index_sequence<std::variant_size_v<Variant>>{});
}

template <typename SigTag>
inline BatchnormSignatureKey makeKeyFromSigTag(std::type_identity<SigTag>)
{
    return BatchnormSignatureKey{
        .inputDataType = SigTag::INPUT_DATA_TYPE,
        .scaleBiasDataType = SigTag::SCALE_BIAS_DATA_TYPE,
        .meanVarianceDataType = SigTag::MEAN_VARIANCE_DATA_TYPE,
        .nodeAttributesType = SigTag::NODE_ATTRIBUTES_TYPE,
    };
}

static TensorVariant& unwrapTensorVariant(std::any& a)
{
    if(auto p = std::any_cast<TensorVariant>(&a))
    {
        return *p;
    }
    if(auto pr = std::any_cast<std::reference_wrapper<TensorVariant>>(&a))
    {
        return pr->get();
    }
    throw std::bad_any_cast();
}

// Build a BatchnormFn for a signature tag type
template <typename SigTag>
inline BatchnormFn makeFnForSigTag(std::type_identity<SigTag>)
{
    return [](std::any& input,
              std::any& scale,
              std::any& bias,
              std::any& mean,
              std::any& variance,
              std::any& output,
              double epsilon) {
        auto& inVar = unwrapTensorVariant(input);
        auto& scVar = unwrapTensorVariant(scale);
        auto& biVar = unwrapTensorVariant(bias);
        auto& meVar = unwrapTensorVariant(mean);
        auto& vaVar = unwrapTensorVariant(variance);
        auto& outVar = unwrapTensorVariant(output);

        using InputT = typename DataTypeToNative<SigTag::INPUT_DATA_TYPE>::type;
        using ScaleBiasT = typename DataTypeToNative<SigTag::SCALE_BIAS_DATA_TYPE>::type;
        using MeanVarianceT = typename DataTypeToNative<SigTag::MEAN_VARIANCE_DATA_TYPE>::type;

        auto& inTensor = *std::get<std::unique_ptr<TensorBase<InputT>>>(inVar);
        auto& scTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasT>>>(scVar);
        auto& biTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasT>>>(biVar);
        auto& meTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceT>>>(meVar);
        auto& vaTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceT>>>(vaVar);
        auto& outTensor = *std::get<std::unique_ptr<TensorBase<InputT>>>(outVar);

        BatchnormBuilder<SigTag{}>::Instance::batchnormFwdInference(
            inTensor, scTensor, biTensor, meTensor, vaTensor, outTensor, epsilon);
    };
}

inline BatchnormFn buildFnFromSignatureVariant(const BatchnormSignatureVariants& v)
{
    return std::visit(
        [](auto sigTag) {
            using Sig = std::decay_t<decltype(sigTag)>;
            return makeFnForSigTag<Sig>(std::type_identity<Sig>{});
        },
        v);
}

// Registration helpers for each supported type
struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        forEachBatchnormSignatureAlt([](auto tag) {
            auto key = makeKeyFromSigTag(tag);
            batchnormRegistry()[key] = makeFnForSigTag(tag);
        });
    }
};
static BatchnormRegistryInitializer _batchnormRegistryInitializer;
}
}
