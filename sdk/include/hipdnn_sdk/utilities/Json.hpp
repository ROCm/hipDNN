#include "batchnorm_backward_attributes_generated.h"
#include "batchnorm_inference_attributes_generated.h"
#include "data_types_generated.h"
#include "pointwise_attributes_generated.h"
#include "tensor_attributes_generated.h"
#include <flatbuffers/flatbuffer_builder.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/hip_fp16_gcc.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <iostream>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <optional>

namespace hipdnn_sdk::json
{
template <class T>
concept JsonConstructible = requires(T obj) {
    { obj } -> std::convertible_to<nlohmann::json>;
};
}

namespace std
{
template <hipdnn_sdk::json::JsonConstructible T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, vector<T> const& vec)
{
    vectorList = nlohmann::json::array();
    for(auto v : vec)
    {
        vectorList.push_back(v);
    }
}

template <hipdnn_sdk::json::JsonConstructible T>
// NOLINTNEXTLINE(readability-identifier-naming)
void from_json(const nlohmann::json& vecJson, vector<T>& vec)
{
    if(!vecJson.is_array())
    {
        throw std::runtime_error("from_json: Attempting to deserialize non-array into vector");
    }
    vec.reserve(vecJson.size());
    for(const auto& v : vecJson)
    {
        vec.push_back(v.get<T>());
    }
}

// template <hipdnn_sdk::json::JsonConstructible T>
// // NOLINTNEXTLINE(readability-identifier-naming)
// void to_json(nlohmann::json& entry, optional<T> const& opt)
// {
//     if(opt.has_value())
//     {
//         entry = opt.value();
//     }
// }

template <hipdnn_sdk::json::JsonConstructible T>
// NOLINTNEXTLINE(readability-identifier-naming)
void from_json(const nlohmann::json& entry, optional<T>& opt)
{
    opt = (entry.is_null()) ? std::nullopt : std::optional<T>{entry.get<T>()};
}

}

namespace flatbuffers
{
template <hipdnn_sdk::json::JsonConstructible T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, Vector<Offset<T>> const& vec)
{
    vectorList = nlohmann::json::array();
    for(auto v : vec)
    {
        vectorList.push_back(*v);
    }
}

}

namespace hipdnn_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(
    NodeAttributes,
    {{NodeAttributes::BatchnormInferenceAttributes, "BatchnormInferenceAttributes"},
     {NodeAttributes::PointwiseAttributes, "PointwiseAttributes"},
     {NodeAttributes::BatchnormBackwardAttributes, "BatchnormBackwardAttributes"},
     {NodeAttributes::BatchnormAttributes, "BatchnormAttributes"},
     {NodeAttributes::ConvolutionFwdAttributes, "ConvolutionFwdAttributes"},
     {NodeAttributes::NONE, ""}})

NLOHMANN_JSON_SERIALIZE_ENUM(DataType,
                             {
                                 {DataType::UNSET, "unset"},
                                 {DataType::FLOAT, "float"},
                                 {DataType::HALF, "half"},
                                 {DataType::BFLOAT16, "bfloat16"},
                                 {DataType::DOUBLE, "double"},
                                 {DataType::UINT8, "uint8"},
                                 {DataType::INT32, "int32"},
                             }

)

NLOHMANN_JSON_SERIALIZE_ENUM(PointwiseMode,
                             {{PointwiseMode::UNSET, "unset"},
                              {PointwiseMode::ABS, "abs"},
                              {PointwiseMode::ADD, "add"},
                              {PointwiseMode::ADD_SQUARE, "add_square"},
                              {PointwiseMode::BINARY_SELECT, "binary_select"},
                              {PointwiseMode::CEIL, "ceil"},
                              {PointwiseMode::CMP_EQ, "cmp_eq"},
                              {PointwiseMode::CMP_GE, "cmp_ge"},
                              {PointwiseMode::CMP_GT, "cmp_gt"},
                              {PointwiseMode::CMP_LE, "cmp_le"},
                              {PointwiseMode::CMP_LT, "cmp_lt"},
                              {PointwiseMode::CMP_NEQ, "cmp_neq"},
                              {PointwiseMode::DIV, "div"},
                              {PointwiseMode::ELU_BWD, "elu_bwd"},
                              {PointwiseMode::ELU_FWD, "elu_fwd"},
                              {PointwiseMode::ERF, "erf"},
                              {PointwiseMode::EXP, "exp"},
                              {PointwiseMode::FLOOR, "floor"},
                              {PointwiseMode::GELU_APPROX_TANH_BWD, "gelu_approx_tanh_bwd"},
                              {PointwiseMode::GELU_APPROX_TANH_FWD, "gelu_approx_tanh_fwd"},
                              {PointwiseMode::GELU_BWD, "gelu_bwd"},
                              {PointwiseMode::GELU_FWD, "gelu_fwd"},
                              {PointwiseMode::GEN_INDEX, "gen_index"},
                              {PointwiseMode::IDENTITY, "identity"},
                              {PointwiseMode::LOG, "log"},
                              {PointwiseMode::LOGICAL_AND, "logical_and"},
                              {PointwiseMode::LOGICAL_NOT, "logical_not"},
                              {PointwiseMode::LOGICAL_OR, "logical_or"},
                              {PointwiseMode::MAX_OP, "max_op"}, // Max is reserved
                              {PointwiseMode::MIN_OP, "min_op"}, // Min is reserved
                              {PointwiseMode::MUL, "mul"},
                              {PointwiseMode::NEG, "neg"},
                              {PointwiseMode::RECIPROCAL, "reciprocal"},
                              {PointwiseMode::RELU_BWD, "relu_bwd"},
                              {PointwiseMode::RELU_FWD, "relu_fwd"},
                              {PointwiseMode::RSQRT, "rsqrt"},
                              {PointwiseMode::SIGMOID_BWD, "sigmoid_bwd"},
                              {PointwiseMode::SIGMOID_FWD, "sigmoid_fwd"},
                              {PointwiseMode::SIN, "sin"},
                              {PointwiseMode::SOFTPLUS_BWD, "softplus_bwd"},
                              {PointwiseMode::SOFTPLUS_FWD, "softplus_fwd"},
                              {PointwiseMode::SQRT, "sqrt"},
                              {PointwiseMode::SUB, "sub"},
                              {PointwiseMode::SWISH_BWD, "swish_bwd"},
                              {PointwiseMode::SWISH_FWD, "swish_fwd"},
                              {PointwiseMode::TAN, "tan"},
                              {PointwiseMode::TANH_BWD, "tanh_bwd"},
                              {PointwiseMode::TANH_FWD, "tanh_fwd"}})

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& tensorAttrJson, data_objects::TensorAttributes const& tensorAttr)
{
    tensorAttrJson["uid"] = tensorAttr.uid();
    tensorAttrJson["data_type"] = tensorAttr.data_type();
    tensorAttrJson["dims"] = *tensorAttr.dims();
    tensorAttrJson["strides"] = *tensorAttr.strides();
    tensorAttrJson["name"] = tensorAttr.name()->c_str();
    tensorAttrJson["virtual"] = tensorAttr.virtual_();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& batchnormJson, BatchnormInferenceAttributes const& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["x"] = bn.x_tensor_uid();
    inputs["mean"] = bn.mean_tensor_uid();
    inputs["scale"] = bn.scale_tensor_uid();
    inputs["inv_variance"] = bn.inv_variance_tensor_uid();
    inputs["bias"] = bn.bias_tensor_uid();

    batchnormJson["outputs"]["y"] = bn.y_tensor_uid();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& batchnormJson, BatchnormBackwardAttributes const& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["dy"] = bn.dy_tensor_uid();
    inputs["x"] = bn.x_tensor_uid();
    inputs["mean"] = bn.mean_tensor_uid();
    inputs["inv_variance"] = bn.inv_variance_tensor_uid();
    inputs["scale"] = bn.scale_tensor_uid();
    inputs["peer_stats"] = *bn.peer_stats_tensor_uid();

    auto& outputs = batchnormJson["outputs"] = {};
    outputs["dbias"] = bn.dbias_tensor_uid();
    outputs["dscale"] = bn.dscale_tensor_uid();
    outputs["dx"] = bn.dx_tensor_uid();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& batchnormJson, BatchnormAttributes const& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};
    auto& outputs = batchnormJson["outputs"] = {};

    inputs["x"] = bn.x_tensor_uid();
    inputs["scale"] = bn.scale_tensor_uid();
    inputs["bias"] = bn.bias_tensor_uid();
    inputs["epsilon"] = bn.epsilon_tensor_uid();
    inputs["peer_stats"] = *bn.peer_stats_tensor_uid();
    inputs["prev_running_mean"] = bn.prev_running_mean_tensor_uid();
    inputs["prev_running_variance"] = bn.prev_running_variance_tensor_uid();
    inputs["momentum"] = bn.momentum_tensor_uid();

    outputs["y"] = bn.y_tensor_uid();
    outputs["mean"] = bn.mean_tensor_uid();
    outputs["inv_variance"] = bn.inv_variance_tensor_uid();
    outputs["next_running_mean"] = bn.next_running_mean_tensor_uid();
    outputs["next_running_variance"] = bn.next_running_variance_tensor_uid();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& pointwiseJson, PointwiseAttributes const& pw)
{
    auto& inputs = pointwiseJson["inputs"] = {};

    inputs["operation"] = pw.operation();
    inputs["relu_lower_clip"] = pw.relu_lower_clip();
    inputs["relu_upper_clip"] = pw.relu_upper_clip();
    inputs["relu_lower_slope"] = pw.relu_lower_slope();
    inputs["axis_tensor_uid"] = pw.axis_tensor_uid();
    inputs["in_0_tensor_uid"] = pw.in_0_tensor_uid();
    inputs["in_1_tensor_uid"] = pw.in_1_tensor_uid();
    inputs["in_2_tensor_uid"] = pw.in_2_tensor_uid();

    pointwiseJson["outputs"]["out_0_tensor_uid"] = pw.out_0_tensor_uid();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& nodeJson, data_objects::Node const& node)
{
    auto type = node.attributes_type();

    switch(type)
    {
    case data_objects::NodeAttributes::BatchnormInferenceAttributes:
        nodeJson = *node.attributes_as_BatchnormInferenceAttributes();
        break;
    case data_objects::NodeAttributes::BatchnormBackwardAttributes:
        nodeJson = *node.attributes_as_BatchnormBackwardAttributes();
        break;
    case data_objects::NodeAttributes::BatchnormAttributes:
        nodeJson = *node.attributes_as_BatchnormAttributes();
        break;
    case data_objects::NodeAttributes::PointwiseAttributes:
        nodeJson = *node.attributes_as_PointwiseAttributes();
        break;
    default:
        throw std::runtime_error(
            "hipdnn_sdk::data_objects::to_json(Node): Unsupported NodeAttributes type: "
            + std::to_string(static_cast<int8_t>(node.attributes_type())));
    }
    nodeJson["name"] = node.name()->c_str();
    nodeJson["type"] = node.attributes_type();
}

// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& graphJson, data_objects::Graph const& graph)
{
    graphJson["nodes"] = *graph.nodes();
    graphJson["compute_type"] = graph.compute_type();
    graphJson["io_type"] = graph.io_type();
    graphJson["intermediate_type"] = graph.intermediate_type();
    graphJson["name"] = graph.name()->c_str();
    graphJson["tensors"] = *graph.tensors();
}
}

namespace hipdnn_sdk::json
{

template <class T, class Key>
std::optional<T> optionalValue(nlohmann::json obj, Key&& key)
{
    auto it = obj.find(std::forward<Key>(key));
    return (it != obj.end()) ? std::optional<T>(it->template get<T>()) : std::nullopt;
}

template <class T>
auto to(flatbuffers::FlatBufferBuilder& builder, nlohmann::json const& entry);

namespace details
{
template <class T>
struct To
{
};

template <class T, class Allocator>
struct To<std::vector<flatbuffers::Offset<T>, Allocator>>
{
    auto operator()(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& jsonValue)
    {
        if(!jsonValue.is_array())
        {
            throw std::runtime_error("hipdnn_sdk::json::to<vector<T>>(): field is not an array");
        }
        std::vector<flatbuffers::Offset<T>> ret;
        for(const auto& v : jsonValue)
        {
            ret.push_back(to<T>(builder, v));
        }

        return ret;
    }
};
}

template <class T>
auto toVector(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    if(!entry.is_array())
    {
        throw std::runtime_error("hipdnn_sdk::json::to<vector<T>>(): field is not an array");
    }
    std::vector<flatbuffers::Offset<T>> ret;
    for(const auto& v : entry)
    {
        ret.push_back(to<T>(builder, v));
    }

    return ret;
}

template <>
auto to<data_objects::BatchnormInferenceAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                    const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];
    return data_objects::CreateBatchnormInferenceAttributes(
        builder,
        inputs.at("x").get<int64_t>(),
        inputs.at("mean").get<int64_t>(),
        inputs.at("inv_variance").get<int64_t>(),
        inputs.at("scale").get<int64_t>(),
        inputs.at("bias").get<int64_t>(),
        entry.at("outputs").at("y").get<int64_t>());
}

template <>
auto to<data_objects::BatchnormBackwardAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    auto peerStats = inputs["peer_stats"].get<std::vector<int64_t>>();

    return data_objects::CreateBatchnormBackwardAttributesDirect(
        builder,
        inputs.at("dy").get<int64_t>(),
        inputs.at("x").get<int64_t>(),
        inputs.at("mean").get<std::optional<int64_t>>(),
        inputs.at("inv_variance").get<std::optional<int64_t>>(),
        inputs.at("scale").get<int64_t>(),
        &peerStats,
        outputs.at("dx").get<int64_t>(),
        outputs.at("dscale").get<int64_t>(),
        outputs.at("dbias").get<int64_t>());
}

template <>
auto to<data_objects::PointwiseAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                           const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreatePointwiseAttributes(
        builder,
        inputs.at("operation").get<PointwiseMode>(),
        inputs.at("relu_lower_clip").get<std::optional<float>>(),
        inputs.at("relu_upper_clip").get<std::optional<float>>(),
        inputs.at("relu_lower_slope").get<std::optional<float>>(),
        inputs.at("axis_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("in_0_tensor_uid").get<int64_t>(),
        inputs.at("in_1_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("in_2_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("out_0_tensor_uid").get<int64_t>());
}

template <>
auto to<data_objects::BatchnormAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                           const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    auto peerStats = inputs["peer_stats"].get<std::vector<int64_t>>();

    return data_objects::CreateBatchnormAttributesDirect(
        builder,
        inputs.at("x").get<int64_t>(),
        inputs.at("scale").get<int64_t>(),
        inputs.at("bias").get<int64_t>(),
        inputs.at("epsilon").get<int64_t>(),
        &peerStats,
        inputs.at("prev_running_mean").get<std::optional<int64_t>>(),
        inputs.at("prev_running_variance").get<std::optional<int64_t>>(),
        inputs.at("momentum").get<std::optional<int64_t>>(),
        outputs.at("y").get<int64_t>(),
        outputs.at("mean").get<std::optional<int64_t>>(),
        outputs.at("inv_variance").get<std::optional<int64_t>>(),
        outputs.at("next_running_mean").get<std::optional<int64_t>>(),
        outputs.at("next_running_variance").get<std::optional<int64_t>>());
}

template <>
auto to<data_objects::Node>(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    auto type = entry.at("type").get<data_objects::NodeAttributes>();
    auto name = entry.at("name").get<std::string>();

    flatbuffers::Offset<void> node = [&]() {
        switch(type)
        {
        case data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return to<data_objects::BatchnormInferenceAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return to<data_objects::BatchnormBackwardAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BatchnormAttributes:
            return to<data_objects::BatchnormAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::PointwiseAttributes:
            return to<data_objects::PointwiseAttributes>(builder, entry).Union();
        default:
            throw std::runtime_error(
                "hipdnn_sdk::json::to<data_objects::Node>(): Unsupported NodeAttributes type: "
                + std::string{EnumNameNodeAttributes(type)});
        }
    }();

    return data_objects::CreateNodeDirect(builder, name.c_str(), type, node);
}

template <>
auto to<data_objects::TensorAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                        const nlohmann::json& entry)
{
    auto uid = entry.at("uid").get<int64_t>();
    auto name = entry.at("name").get<std::string>();
    auto dataType = entry.at("data_type").get<data_objects::DataType>();
    auto dims = entry.at("dims").get<std::vector<int64_t>>();
    auto strides = entry.at("strides").get<std::vector<int64_t>>();
    bool isVirtual = entry.at("virtual").get<bool>();

    return data_objects::CreateTensorAttributesDirect(
        builder, uid, name.c_str(), dataType, &strides, &dims, isVirtual);
}

template <>
auto to<data_objects::Graph>(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    using namespace data_objects;
    using namespace flatbuffers;

    auto name = entry.at("name").get<std::string>();
    auto computeType = entry.at("compute_type").get<data_objects::DataType>();
    auto ioType = entry.at("io_type").get<data_objects::DataType>();
    auto intermediateType = entry.at("intermediate_type").get<data_objects::DataType>();

    auto nodes = toVector<Node>(builder, entry.at("nodes"));
    auto tensors = toVector<TensorAttributes>(builder, entry.at("tensors"));
    return data_objects::CreateGraphDirect(
        builder, name.c_str(), computeType, intermediateType, ioType, &tensors, &nodes);
}

}
