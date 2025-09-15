// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/TensorView.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

class CpuReferenceGraphExecutor
{
public:
    CpuReferenceGraphExecutor() = default;
    ~CpuReferenceGraphExecutor() = default;

    template <typename T>
    static std::unique_ptr<TensorBase<T>> createHostOnlyShallowTensor(
        void* ptr, const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        return std::make_unique<TensorView<T>>(ptr, dims, strides);
    }

    static std::vector<int64_t> flatbufferVectorToStd(const ::flatbuffers::Vector<int64_t>* fbVec)
    {
        std::vector<int64_t> result;
        if(fbVec == nullptr)
        {
            return result;
        }
        result.reserve(fbVec->size());
        for(auto v : *fbVec)
        {
            result.push_back(v);
        }
        return result;
    }

    static void executeTheGraph(void* graphBuffer,
                                size_t size,
                                std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap = hipdnn_plugin::GraphWrapper(graphBuffer, size);

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
            if(nodeAttributes != nullptr)
            {
                const auto& tensorMap = graphWrap.getTensorMap();
                auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
                //std::ignore = xTensorAttr;
                BatchnormSignatureKey key{
                    .inputDataType = xTensorAttr->data_type(),
                    .nodeAttributesType
                    = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
                };
                auto it = batchnormRegistry().find(key);
                if(it != batchnormRegistry().end())
                {
                    BatchnormFn fn = it->second;

                    auto shallowXTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(xTensorAttr->uid()),
                        flatbufferVectorToStd(xTensorAttr->dims()),
                        flatbufferVectorToStd(xTensorAttr->strides()));
                    std::any input = std::ref(*shallowXTensor);

                    auto yTensorAttr = tensorMap.at(nodeAttributes->y_tensor_uid());
                    auto shallowYTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(yTensorAttr->uid()),
                        flatbufferVectorToStd(yTensorAttr->dims()),
                        flatbufferVectorToStd(yTensorAttr->strides()));
                    std::any output = std::ref(*shallowYTensor);

                    auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
                    auto shallowScaleTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(scaleTensorAttr->uid()),
                        flatbufferVectorToStd(scaleTensorAttr->dims()),
                        flatbufferVectorToStd(scaleTensorAttr->strides()));
                    std::any scale = std::ref(*shallowScaleTensor);

                    auto biasTensorAttr = tensorMap.at(nodeAttributes->bias_tensor_uid());
                    auto shallowBiasTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(biasTensorAttr->uid()),
                        flatbufferVectorToStd(biasTensorAttr->dims()),
                        flatbufferVectorToStd(biasTensorAttr->strides()));
                    std::any bias = std::ref(*shallowBiasTensor);

                    auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
                    auto shallowMeanTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(meanTensorAttr->uid()),
                        flatbufferVectorToStd(meanTensorAttr->dims()),
                        flatbufferVectorToStd(meanTensorAttr->strides()));
                    std::any mean = std::ref(*shallowMeanTensor);

                    auto invVarianceTensorAttr
                        = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());
                    auto shallowInvVarianceTensor = createHostOnlyShallowTensor<float>(
                        variantPack.at(invVarianceTensorAttr->uid()),
                        flatbufferVectorToStd(invVarianceTensorAttr->dims()),
                        flatbufferVectorToStd(invVarianceTensorAttr->strides()));
                    std::any variance = std::ref(*shallowInvVarianceTensor);

                    fn(input, scale, bias, mean, variance, output, 1e-3);
                }
                else
                {
                    throw std::runtime_error("No registered function for the given signature");
                }
            }
            else
            {
                throw std::runtime_error("Unsupported node attributes type");
            }
        }
    }
};

}
}
