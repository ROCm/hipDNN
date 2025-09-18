#include "batchnorm_inference_attributes_generated.h"
#include "tensor_attributes_generated.h"
#include <flatbuffers/flatbuffer_builder.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/hip_fp16_gcc.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
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
}

namespace flatbuffers
{
template <class T>
    requires(hipdnn_sdk::json::JsonConstructible<T>)
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
void to_json(nlohmann::json& nodeJson, data_objects::Node const& node)
{
    auto type = node.attributes_type();

    if(type == data_objects::NodeAttributes::BatchnormInferenceAttributes)
    {
        nodeJson = nlohmann::json(*node.attributes_as_BatchnormInferenceAttributes());
    }
    else
    {
        throw std::runtime_error("Unsupported NodeAttribute  type: "
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

auto batchnormInferenceAttributes(flatbuffers::FlatBufferBuilder& builder,
                                  const nlohmann::json& attributes)
{
    auto& input = attributes["inputs"];
    return data_objects::CreateBatchnormInferenceAttributes(
        builder,
        input["x"].get<int64_t>(),
        optionalValue<int64_t>(input, "mean"),
        optionalValue<int64_t>(input, "inv_variance"),
        input["scale"].get<int64_t>(),
        input["bias"].get<int64_t>(),
        attributes["outputs"]["y"].get<int64_t>());
}

auto node(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& inNode)
{
    auto type = inNode["type"].get<data_objects::NodeAttributes>();
    auto name = inNode["name"].get<std::string>();

    flatbuffers::Offset<void> node = [&]() {
        switch(type)
        {
        case data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return batchnormInferenceAttributes(builder, inNode).Union();
        default:
            throw std::runtime_error("Unsupported NodeAttribute type: "
                                     + std::string{EnumNameNodeAttributes(type)});
        }
    }();

    return data_objects::CreateNodeDirect(builder, name.c_str(), type, node);
}

auto tensorAttributes(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& tensorAttrJson)
{
    auto uid = tensorAttrJson["uid"].get<int64_t>();
    auto name = tensorAttrJson["name"].get<std::string>();
    auto dataType = tensorAttrJson["data_type"].get<data_objects::DataType>();
    auto dims = tensorAttrJson["dims"].get<std::vector<int64_t>>();
    auto strides = tensorAttrJson["strides"].get<std::vector<int64_t>>();
    bool isVirtual = tensorAttrJson.value<bool>("virtual", false);

    return data_objects::CreateTensorAttributesDirect(
        builder, uid, name.c_str(), dataType, &strides, &dims, isVirtual);
}

auto graph(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& inGraph)
{
    auto name = inGraph.value<std::string>("name", std::string{});
    auto computeType = inGraph["compute_type"].get<data_objects::DataType>();
    auto ioType = inGraph["io_type"].get<data_objects::DataType>();
    auto intermediateType = inGraph["intermediate_type"].get<data_objects::DataType>();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    if(!inGraph["nodes"].is_array())
    {
        throw std::runtime_error("json::graph: \"nodes\" field is not an array");
    }
    for(const auto& n : inGraph["nodes"])
    {
        nodes.push_back(node(builder, n));
    }

    if(!inGraph["tensors"].is_array())
    {
        throw std::runtime_error("json::graph: \"tensors\" field is not an array");
    }
    for(const auto& t : inGraph["tensors"])
    {
        tensors.push_back(tensorAttributes(builder, t));
    }

    return data_objects::CreateGraphDirect(
        builder, name.c_str(), computeType, intermediateType, ioType, &tensors, &nodes);
}
}
