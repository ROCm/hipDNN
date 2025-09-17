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

using BatchnormFwdInferenceFn
    = std::function<void(std::any&, std::any&, std::any&, std::any&, std::any&, std::any&, double)>;

// Registry keyed by BatchnormSignatureKey
inline std::unordered_map<BatchnormSignatureKey, BatchnormFwdInferenceFn>& batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureKey, BatchnormFwdInferenceFn> _reg;
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

template <typename BatchnormSignature>
inline BatchnormSignatureKey makeKeyFromBatchnormSignature(std::type_identity<BatchnormSignature>)
{
    return BatchnormSignatureKey{
        .inputDataType = BatchnormSignature::INPUT_DATA_TYPE,
        .scaleBiasDataType = BatchnormSignature::SCALE_BIAS_DATA_TYPE,
        .meanVarianceDataType = BatchnormSignature::MEAN_VARIANCE_DATA_TYPE,
        .nodeAttributesType = BatchnormSignature::NODE_ATTRIBUTES_TYPE,
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

template <typename BatchnormSignature>
inline BatchnormFwdInferenceFn
    makeBatchnormFwdInferenceFnForSignature(std::type_identity<BatchnormSignature>)
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

        using InputT = typename DataTypeToNative<BatchnormSignature::INPUT_DATA_TYPE>::type;
        using ScaleBiasT =
            typename DataTypeToNative<BatchnormSignature::SCALE_BIAS_DATA_TYPE>::type;
        using MeanVarianceT =
            typename DataTypeToNative<BatchnormSignature::MEAN_VARIANCE_DATA_TYPE>::type;

        auto& inTensor = *std::get<std::unique_ptr<TensorBase<InputT>>>(inVar);
        auto& scTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasT>>>(scVar);
        auto& biTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasT>>>(biVar);
        auto& meTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceT>>>(meVar);
        auto& vaTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceT>>>(vaVar);
        auto& outTensor = *std::get<std::unique_ptr<TensorBase<InputT>>>(outVar);

        BatchnormFwdInferenceBuilder<BatchnormSignature{}>::Instance::batchnormFwdInference(
            inTensor, scTensor, biTensor, meTensor, vaTensor, outTensor, epsilon);
    };
}

inline BatchnormFwdInferenceFn
    buildBatchnormFwdInferenceFnFromSignatureVariant(const BatchnormSignatureVariants& v)
{
    return std::visit(
        [](auto batchnormSignature) {
            using BatchnormSignature = std::decay_t<decltype(batchnormSignature)>;
            return makeBatchnormFwdInferenceFnForSignature<BatchnormSignature>(
                std::type_identity<BatchnormSignature>{});
        },
        v);
}

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        forEachBatchnormSignatureAlt([](auto signature) {
            auto key = makeKeyFromBatchnormSignature(signature);
            batchnormRegistry()[key] = makeBatchnormFwdInferenceFnForSignature(signature);
        });
    }
};

static BatchnormRegistryInitializer _batchnormRegistryInitializer;
}
}
