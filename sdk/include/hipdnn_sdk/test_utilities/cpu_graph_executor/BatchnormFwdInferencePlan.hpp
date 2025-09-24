// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormFwdInferenceParams
{
    BatchnormFwdInferenceParams(
        const hipdnn_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& meanAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& invVarianceAttributes,
        double eps)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , biasTensor(unpackTensorAttributes(biasAttributes))
        , meanTensor(unpackTensorAttributes(meanAttributes))
        , invVarianceTensor(unpackTensorAttributes(invVarianceAttributes))
        , epsilon(eps)
    {
    }

    hipdnn_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_sdk::data_objects::TensorAttributesT meanTensor;
    hipdnn_sdk::data_objects::TensorAttributesT invVarianceTensor;
    double epsilon; //todo, fix this.
};

//todo, move to correct place one structs are in better locations
template <typename T>
inline std::unique_ptr<hipdnn_sdk::utilities::ShallowTensor<T>>
    CreateShallowTensor(const hipdnn_sdk::data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<hipdnn_sdk::utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

class IGraphNodePlanExecutor
{
public:
    virtual ~IGraphNodePlanExecutor() = default;

    virtual void execute(std::unordered_map<int64_t, void*>& variantPack) = 0;
};

class IGraphNodePlanBuilder
{
public:
    virtual ~IGraphNodePlanBuilder() = default;
    virtual bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const
        = 0;

    virtual std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node)
        = 0;
};

template <BatchnormSignatureRegistryKey Key>
class BatchnormFwdPlan : public IGraphNodePlanExecutor
{
public:
    using InputDataType = DataTypeToNative<Key.inputDataType>;
    using ScaleBiasDataType = DataTypeToNative<Key.scaleBiasDataType>;
    using MeanVarianceDataType = DataTypeToNative<Key.meanVarianceDataType>;

    BatchnormFwdPlan(BatchnormFwdInferenceParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor = CreateShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = CreateShallowTensor<InputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = CreateShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto shallowBiasTensor = CreateShallowTensor<ScaleBiasDataType>(
            _params.biasTensor, variantPack.at(_params.biasTensor.uid));

        auto shallowMeanTensor = CreateShallowTensor<MeanVarianceDataType>(
            _params.meanTensor, variantPack.at(_params.meanTensor.uid));

        auto shallowInvVarianceTensor = CreateShallowTensor<MeanVarianceDataType>(
            _params.invVarianceTensor, variantPack.at(_params.invVarianceTensor.uid));

        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormFwdInference(*shallowXTensor,
                                  *shallowScaleTensor,
                                  *shallowBiasTensor,
                                  *shallowMeanTensor,
                                  *shallowInvVarianceTensor,
                                  *shallowYTensor,
                                  _params.epsilon);

        std::cout << "Executed Batchnorm Fwd Inference Plan" << std::endl;
    }

private:
    BatchnormFwdInferenceParams _params;
};

template <BatchnormSignatureRegistryKey Key>
class BatchnormFwdInferencePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = DataTypeToNative<Key.inputDataType>;
    using ScaleBiasDataType = DataTypeToNative<Key.scaleBiasDataType>;
    using MeanVarianceDataType = DataTypeToNative<Key.meanVarianceDataType>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        std::ignore = node;
        std::ignore = tensorMap;

        //do node checks here
        //check type is applicable
        // check bnorm registry has applicable executor.
        //todo
        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BatchnormInferenceAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        BatchnormFwdInferenceParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                           *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                           *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                           *tensorMap.at(nodeAttributes->bias_tensor_uid()),
                                           *tensorMap.at(nodeAttributes->mean_tensor_uid()),
                                           *tensorMap.at(nodeAttributes->inv_variance_tensor_uid()),
                                           1e-3);

        return std::make_unique<BatchnormFwdPlan<Key>>(std::move(params));
    }
};

}
