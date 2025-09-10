// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

enum class DeviceType
{
    CPU = 0,
    GPU = 1
};

class RefImplBase
{
public:
    virtual ~RefImplBase() = default;
    virtual bool isApplicable(const Node& node) = 0;
    virtual void run(const Node& node,
                     const std::unordered_map<int64_t, void*>& variantPack,
                     void* workspace)
        = 0;
    virtual std::string getTypeInfo() const = 0;
};

struct RegistryKey
{
    NodeAttributes op;
    DataType compute_type;
    DataType intermediate_type;
    DataType io_type;
    DeviceType device;

    bool operator==(const RegistryKey& other) const
    {
        return op == other.op && compute_type == other.compute_type
               && intermediate_type == other.intermediate_type && io_type == other.io_type
               && device == other.device;
    }
};

struct RegistryKeyHash
{
    std::size_t operator()(const RegistryKey& k) const
    {
        return (static_cast<std::size_t>(k.op) * 73856093U)
               ^ (static_cast<std::size_t>(k.compute_type) * 19349663U)
               ^ (static_cast<std::size_t>(k.intermediate_type) * 83492791U)
               ^ (static_cast<std::size_t>(k.io_type) * 51439189U)
               ^ (static_cast<std::size_t>(k.device) * 97844151U);
    }
};

template <DataType DT>
struct DataTypeToType;

template <>
struct DataTypeToType<DataType_FLOAT>
{
    using type = float;
};
template <>
struct DataTypeToType<DataType_DOUBLE>
{
    using type = double;
};

template <NodeAttributes Op, typename ComputeT, typename IntermediateT, typename IoT>
struct CpuImplTraits;

#define DEFINE_CPU_IMPL_TRAITS(Op, ImplClass)                          \
    template <typename ComputeT, typename IntermediateT, typename IoT> \
    struct CpuImplTraits<Op, ComputeT, IntermediateT, IoT>             \
    {                                                                  \
        using type = ImplClass<ComputeT, IntermediateT, IoT>;          \
    };

DEFINE_CPU_IMPL_TRAITS(NodeAttributes_ConvolutionFwdAttributes,
                       hipdnn_sdk::reference_test_utilities::CpuFpReferenceConvolutionImpl)
DEFINE_CPU_IMPL_TRAITS(NodeAttributes_BatchnormInferenceAttributes,
                       hipdnn_sdk::test_utilities::CpuFpReferenceBatchnormImpl)
DEFINE_CPU_IMPL_TRAITS(NodeAttributes_BatchnormBackwardAttributes,
                       hipdnn_sdk::test_utilities::CpuFpReferenceBatchnormImpl)

#undef DEFINE_CPU_IMPL_TRAITS

template <NodeAttributes Op,
          DataType ComputeType,
          DataType IntermediateType,
          DataType IoType,
          template <NodeAttributes, typename, typename, typename> class ImplTraits>
class RefImplWrapper : public RefImplBase
{
    using ComputeT = typename DataTypeToType<ComputeType>::type;
    using IntermediateT = typename DataTypeToType<IntermediateType>::type;
    using IoT = typename DataTypeToType<IoType>::type;
    using ImplType = typename ImplTraits<Op, ComputeT, IntermediateT, IoT>::type;

public:
    bool isApplicable(const Node& node) override
    {
        if(node.attributes_type() != Op)
        {
            return false;
        }

        return ImplType::isApplicable(node);
    }

    void run(const Node& node,
             const std::unordered_map<int64_t, void*>& variantPack,
             void* workspace) override
    {
        (void)workspace; // Unused for now

        if constexpr(Op == NodeAttributes_ConvolutionFwdAttributes)
        {
            std::cout << "Executing ConvolutionFwd<" << typeid(ComputeT).name() << ","
                      << typeid(IntermediateT).name() << "," << typeid(IoT).name() << "> with "
                      << variantPack.size() << " tensors\n";

            auto* conv_attrs = node.attributes_as_ConvolutionFwdAttributes();
            int64_t x_uid = conv_attrs->x_tensor_uid();
            int64_t w_uid = conv_attrs->w_tensor_uid();
            int64_t y_uid = conv_attrs->y_tensor_uid();

            auto& input = *static_cast<Tensor<IoT>*>(variantPack.at(x_uid));
            auto& weight = *static_cast<Tensor<IoT>*>(variantPack.at(w_uid));
            auto& output = *static_cast<Tensor<IoT>*>(variantPack.at(y_uid));

            std::vector<int64_t> strides(conv_attrs->stride()->begin(),
                                         conv_attrs->stride()->end());
            std::vector<int64_t> dilations(conv_attrs->dilation()->begin(),
                                           conv_attrs->dilation()->end());

            std::vector<int64_t> padding;
            auto* pre_pad = conv_attrs->pre_padding();
            auto* post_pad = conv_attrs->post_padding();
            if(pre_pad && post_pad)
            {
                padding.reserve(pre_pad->size() + post_pad->size());
                padding.insert(padding.end(), pre_pad->begin(), pre_pad->end());
                padding.insert(padding.end(), post_pad->begin(), post_pad->end());
            }

            ImplType::convFwdInference(input, weight, output, strides, dilations, padding);
        }
        else if constexpr(Op == NodeAttributes_BatchnormInferenceAttributes)
        {
            std::cout << "Executing BatchnormInference<" << typeid(ComputeT).name() << ","
                      << typeid(IntermediateT).name() << "," << typeid(IoT).name() << "> with "
                      << variantPack.size() << " tensors\n";

            auto* bn_attrs = node.attributes_as_BatchnormInferenceAttributes();
            int64_t x_uid = bn_attrs->x_tensor_uid();
            int64_t scale_uid = bn_attrs->scale_tensor_uid();
            int64_t bias_uid = bn_attrs->bias_tensor_uid();
            int64_t y_uid = bn_attrs->y_tensor_uid();

            auto mean_uid_opt = bn_attrs->mean_tensor_uid();
            auto var_uid_opt = bn_attrs->inv_variance_tensor_uid();

            if(!mean_uid_opt.has_value() || !var_uid_opt.has_value())
            {
                throw std::runtime_error(
                    "BatchnormInference requires both mean and variance tensor UIDs");
            }

            int64_t mean_uid = mean_uid_opt.value();
            int64_t var_uid = var_uid_opt.value();

            auto& input = *static_cast<Tensor<IoT>*>(variantPack.at(x_uid));
            auto& scale = *static_cast<Tensor<IntermediateT>*>(variantPack.at(scale_uid));
            auto& bias = *static_cast<Tensor<IntermediateT>*>(variantPack.at(bias_uid));
            auto& estimatedMean = *static_cast<Tensor<IntermediateT>*>(variantPack.at(mean_uid));
            auto& estimatedVar = *static_cast<Tensor<IntermediateT>*>(variantPack.at(var_uid));
            auto& output = *static_cast<Tensor<IoT>*>(variantPack.at(y_uid));

            constexpr double epsilon = 1e-5;

            ImplType::batchnormFwdInference(
                input, scale, bias, estimatedMean, estimatedVar, output, epsilon);
        }
        else if constexpr(Op == NodeAttributes_BatchnormBackwardAttributes)
        {
            std::cout << "Executing BatchnormBackward<" << typeid(ComputeT).name() << ","
                      << typeid(IntermediateT).name() << "," << typeid(IoT).name() << "> with "
                      << variantPack.size() << " tensors\n";

            auto* bn_bwd_attrs = node.attributes_as_BatchnormBackwardAttributes();
            int64_t dy_uid = bn_bwd_attrs->dy_tensor_uid();
            int64_t x_uid = bn_bwd_attrs->x_tensor_uid();
            int64_t scale_uid = bn_bwd_attrs->scale_tensor_uid();
            int64_t dx_uid = bn_bwd_attrs->dx_tensor_uid();
            int64_t dscale_uid = bn_bwd_attrs->dscale_tensor_uid();
            int64_t dbias_uid = bn_bwd_attrs->dbias_tensor_uid();

            auto mean_uid_opt = bn_bwd_attrs->mean_tensor_uid();
            auto inv_var_uid_opt = bn_bwd_attrs->inv_variance_tensor_uid();

            if(!mean_uid_opt.has_value() || !inv_var_uid_opt.has_value())
            {
                throw std::runtime_error(
                    "BatchnormBackward requires both mean and inv_variance tensor UIDs");
            }

            int64_t mean_uid = mean_uid_opt.value();
            int64_t inv_var_uid = inv_var_uid_opt.value();

            auto& dy = *static_cast<Tensor<IoT>*>(variantPack.at(dy_uid));
            auto& x = *static_cast<Tensor<IoT>*>(variantPack.at(x_uid));
            auto& mean = *static_cast<Tensor<IntermediateT>*>(variantPack.at(mean_uid));
            auto& invVariance = *static_cast<Tensor<IntermediateT>*>(variantPack.at(inv_var_uid));
            auto& scale = *static_cast<Tensor<IntermediateT>*>(variantPack.at(scale_uid));
            auto& dx = *static_cast<Tensor<IoT>*>(variantPack.at(dx_uid));
            auto& dscale = *static_cast<Tensor<IntermediateT>*>(variantPack.at(dscale_uid));
            auto& dbias = *static_cast<Tensor<IntermediateT>*>(variantPack.at(dbias_uid));

            ImplType::batchnormBwd(dy, x, mean, invVariance, scale, dx, dscale, dbias);
        }
    }

    std::string getTypeInfo() const override
    {
        std::string opName;
        switch(Op)
        {
        case NodeAttributes_ConvolutionFwdAttributes:
            opName = "ConvolutionFwd";
            break;
        case NodeAttributes_BatchnormInferenceAttributes:
            opName = "BatchnormInference";
            break;
        case NodeAttributes_BatchnormBackwardAttributes:
            opName = "BatchnormBackward";
            break;
        default:
            opName = "Unknown";
            break;
        }
        return opName + "<" + typeid(ComputeT).name() + "," + typeid(IntermediateT).name() + ","
               + typeid(IoT).name() + ">";
    }
};

class RefImplRegistry
{
    using Map = std::unordered_map<RegistryKey, std::unique_ptr<RefImplBase>, RegistryKeyHash>;
    Map registry;

public:
    template <NodeAttributes Op, DataType ComputeType, DataType IntermediateType, DataType IoType>
    void registerCpuImpl()
    {
        RegistryKey key{Op, ComputeType, IntermediateType, IoType, DeviceType::CPU};
        registry[key] = std::make_unique<
            RefImplWrapper<Op, ComputeType, IntermediateType, IoType, CpuImplTraits>>();
    }

    template <NodeAttributes Op,
              DataType ComputeType,
              DataType IntermediateType,
              DataType IoType,
              template <NodeAttributes, typename, typename, typename> class ImplTraits>
    void registerImpl(DeviceType device)
    {
        RegistryKey key{Op, ComputeType, IntermediateType, IoType, device};
        registry[key] = std::make_unique<
            RefImplWrapper<Op, ComputeType, IntermediateType, IoType, ImplTraits>>();
    }

    RefImplBase* get(NodeAttributes op,
                     DataType compute_type,
                     DataType intermediate_type,
                     DataType io_type,
                     DeviceType device)
    {
        RegistryKey key{op, compute_type, intermediate_type, io_type, device};
        auto it = registry.find(key);
        if(it == registry.end())
        {
            return nullptr;
        }
        return it->second.get();
    }

    RefImplBase*
        get(NodeAttributes op, DataType compute_type, DataType intermediate_type, DataType io_type)
    {
        return get(op, compute_type, intermediate_type, io_type, DeviceType::CPU);
    }

    void printRegistered() const
    {
        std::cout << "Registered implementations (" << registry.size() << " total):\n";
        for(const auto& [key, impl] : registry)
        {
            std::string deviceStr = (key.device == DeviceType::CPU) ? "CPU" : "GPU";
            std::cout << "  [" << deviceStr << "] " << impl->getTypeInfo() << "\n";
        }
    }

    size_t size() const
    {
        return registry.size();
    }
};

namespace auto_registration
{
template <DataType ComputeType, DataType IntermediateType, DataType IoType>
struct TypeCombination
{
};

using SupportedTypeCombinations = std::tuple<
    TypeCombination<DataType_FLOAT, DataType_FLOAT, DataType_FLOAT>, // FP32/FP32/FP32
    TypeCombination<DataType_DOUBLE, DataType_DOUBLE, DataType_DOUBLE> // FP64/FP64/FP64
    >;

constexpr NodeAttributes SupportedOps[] = {NodeAttributes_ConvolutionFwdAttributes,
                                           NodeAttributes_BatchnormInferenceAttributes,
                                           NodeAttributes_BatchnormBackwardAttributes};

constexpr DeviceType SupportedDevices[] = {
    DeviceType::CPU,
    // DeviceType::GPU  // Uncomment when GPU implementations available
};

template <NodeAttributes Op, DataType CT, DataType IT, DataType IOT, DeviceType Device>
void register_combination_for_device(RefImplRegistry& reg)
{
    if constexpr(Device == DeviceType::CPU)
    {
        reg.registerImpl<Op, CT, IT, IOT, CpuImplTraits>(Device);
    }
    // Future GPU support:
    // else if constexpr (Device == DeviceType::GPU) {
    //     reg.registerImpl<Op, CT, IT, IOT, GpuImplTraits>(Device);
    // }
}

template <NodeAttributes Op, DeviceType Device, typename... TypeCombinations>
void register_all_types_for_device(RefImplRegistry& reg, std::tuple<TypeCombinations...>)
{
    auto register_tuple
        = [&reg]<DataType CT, DataType IT, DataType IOT>(TypeCombination<CT, IT, IOT>) {
              register_combination_for_device<Op, CT, IT, IOT, Device>(reg);
          };

    (register_tuple(TypeCombinations{}), ...);
}

template <NodeAttributes Op>
void register_operation_for_all_devices(RefImplRegistry& reg)
{
    for(DeviceType device : SupportedDevices)
    {
        if(device == DeviceType::CPU)
        {
            register_all_types_for_device<Op, DeviceType::CPU>(reg, SupportedTypeCombinations{});
        }
        // Future: add GPU case when available
    }
}

inline void auto_register(RefImplRegistry& reg)
{
    for(NodeAttributes op : SupportedOps)
    {
        if(op == NodeAttributes_ConvolutionFwdAttributes)
        {
            register_operation_for_all_devices<NodeAttributes_ConvolutionFwdAttributes>(reg);
        }
        else if(op == NodeAttributes_BatchnormInferenceAttributes)
        {
            register_operation_for_all_devices<NodeAttributes_BatchnormInferenceAttributes>(reg);
        }
        else if(op == NodeAttributes_BatchnormBackwardAttributes)
        {
            register_operation_for_all_devices<NodeAttributes_BatchnormBackwardAttributes>(reg);
        }
    }
}
}

inline RefImplRegistry& getGlobalRegistry()
{
    static RefImplRegistry instance;
    static bool initialized = false;

    if(!initialized)
    {
        auto_registration::auto_register(instance);
        initialized = true;
    }

    return instance;
}

inline void executeOperation(NodeAttributes op,
                             DataType compute_type,
                             DataType intermediate_type,
                             DataType io_type,
                             const Node& node,
                             const std::unordered_map<int64_t, void*>& variantPack,
                             void* workspace = nullptr)
{
    auto& registry = getGlobalRegistry();
    auto* impl = registry.get(op, compute_type, intermediate_type, io_type);

    if(!impl)
    {
        throw std::runtime_error("No implementation found for operation type: "
                                 + std::to_string(static_cast<int>(op))
                                 + " with types: " + std::to_string(static_cast<int>(compute_type))
                                 + "/" + std::to_string(static_cast<int>(intermediate_type)) + "/"
                                 + std::to_string(static_cast<int>(io_type)));
    }

    if(!impl->isApplicable(node))
    {
        throw std::runtime_error("Implementation doesn't support this node configuration: "
                                 + impl->getTypeInfo());
    }

    impl->run(node, variantPack, workspace);
}

template <template <NodeAttributes, typename, typename, typename> class ImplTraits>
class ReferenceContainer
{
public:
    template <typename InputT, typename ScaleBiasT, typename MeanVarT>
    void batchnormFwdInference(const TensorBase<InputT>& input,
                               const TensorBase<ScaleBiasT>& scale,
                               const TensorBase<ScaleBiasT>& bias,
                               const TensorBase<MeanVarT>& estimatedMean,
                               const TensorBase<MeanVarT>& estimatedVariance,
                               TensorBase<InputT>& output,
                               double epsilon)
    {
        using ImplType = typename ImplTraits<NodeAttributes_BatchnormInferenceAttributes,
                                             InputT,
                                             ScaleBiasT,
                                             MeanVarT>::type;
        ImplType::batchnormFwdInference(
            input, scale, bias, estimatedMean, estimatedVariance, output, epsilon);
    }

    template <typename InputT, typename ScaleBiasT, typename MeanVarT>
    void batchnormBwd(const TensorBase<InputT>& dy,
                      const TensorBase<InputT>& x,
                      const TensorBase<MeanVarT>& bnMean,
                      const TensorBase<MeanVarT>& bnInvVariance,
                      const TensorBase<ScaleBiasT>& bnScale,
                      TensorBase<InputT>& dx,
                      TensorBase<ScaleBiasT>& dScale,
                      TensorBase<ScaleBiasT>& dBias)
    {
        using ImplType = typename ImplTraits<NodeAttributes_BatchnormBackwardAttributes,
                                             InputT,
                                             ScaleBiasT,
                                             MeanVarT>::type;
        ImplType::batchnormBwd(dy, x, bnMean, bnInvVariance, bnScale, dx, dScale, dBias);
    }

    template <typename T>
    void convFwdInference(const TensorBase<T>& input,
                          const TensorBase<T>& weight,
                          TensorBase<T>& output,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          const std::vector<int64_t>& padding)
    {
        using ImplType =
            typename ImplTraits<NodeAttributes_ConvolutionFwdAttributes, T, T, T>::type;
        ImplType::convFwdInference(input, weight, output, strides, dilations, padding);
    }
};

// Convenience type aliases
using CpuReferenceContainer = ReferenceContainer<CpuImplTraits>;

// Future GPU support example:
// template<NodeAttributes Op, typename ComputeT, typename IntermediateT, typename IoT>
// struct GpuImplTraits;
// using GpuReferenceContainer = ReferenceContainer<GpuImplTraits>;

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
