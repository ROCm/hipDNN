#include "batchnorm_attributes_generated.h"
#include "batchnorm_inference_attributes_generated.h"
#include <flatbuffers/flatbuffer_builder.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <iostream>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>

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
}

namespace hipdnn_sdk::json
{

// template<class T>
// nlohmann::json json(std::vector<T> )

// // NOLINT(readability-identifier=)
// nlohmann::json to_json(data_objects::TensorAttributes const& attr)
// {
//     nlohmann::json attrJson;

//     attrJson["uid"] = attr.uid();
//     attrJson["data_type"] = attr.data_type();
//     attrJson["dims"] = attr.dims();
// }

nlohmann::json json(data_objects::BatchnormInferenceAttributes const& bn)
{
    nlohmann::json batchnormJson;
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["x"] = bn.x_tensor_uid();
    inputs["mean"] = bn.mean_tensor_uid();
    inputs["scale"] = bn.scale_tensor_uid();
    inputs["inv_variance"] = bn.inv_variance_tensor_uid();
    inputs["bias"] = bn.bias_tensor_uid();

    batchnormJson["outputs"]["y"] = bn.y_tensor_uid();

    return batchnormJson;
}

nlohmann::json json(data_objects::Node const& node)
{
    auto type = node.attributes_type();
    nlohmann::json nodeJson = [&]() {
        if(type == data_objects::NodeAttributes::BatchnormInferenceAttributes)
        {
            return json(*node.attributes_as_BatchnormInferenceAttributes());
        }

        throw std::runtime_error("Unsupported NodeAttribute  type: "
                                 + std::to_string(static_cast<int8_t>(node.attributes_type())));
    }();
    nodeJson["name"] = node.name()->c_str();
    nodeJson["type"] = node.attributes_type();

    return nodeJson;
}

nlohmann::json json(data_objects::DataType const& type)
{
    return static_cast<int8_t>(type);
}

nlohmann::json json(data_objects::Graph const& graph)
{

    nlohmann::json graphJson;
    graphJson["nodes"] = nlohmann::json::array();

    for(auto node : *graph.nodes())
    {
        graphJson["nodes"].push_back(json(*node));
    }

    graphJson["compute_type"] = graph.compute_type();
    graphJson["io_type"] = graph.io_type();
    graphJson["intermediate_type"] = graph.intermediate_type();

    graphJson["name"] = graph.name()->c_str();

    // TODO: Handle tensors

    return graphJson;
}

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

auto graph(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& inGraph)
{
    auto name = inGraph.value<std::string>("name", std::string{});
    auto computeType = inGraph["compute_type"].get<data_objects::DataType>();
    auto ioType = inGraph["io_type"].get<data_objects::DataType>();
    auto intermediateType = inGraph["intermediate_type"].get<data_objects::DataType>();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(!inGraph["nodes"].is_array())
    {
        throw std::runtime_error("json::graph: nodes field is not an array");
    }
    for(const auto& n : inGraph["nodes"])
    {
        nodes.push_back(node(builder, n));
    }

    return data_objects::CreateGraphDirect(
        builder, name.c_str(), computeType, intermediateType, ioType, nullptr, &nodes);
}
}
