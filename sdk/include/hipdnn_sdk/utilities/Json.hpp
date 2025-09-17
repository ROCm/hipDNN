#include "batchnorm_attributes_generated.h"
#include "batchnorm_inference_attributes_generated.h"
#include <flatbuffers/flatbuffer_builder.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>

namespace hipdnn_sdk::json
{

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

// data_objects::NodeAttributes nodeAttribute(std::string attr)
// {
//     using namespace data_objects;
//     if(attr == EnumNameNodeAttributes(NodeAttributes_BatchnormInferenceAttributes))
//     {
//         return NodeAttributes_BatchnormInferenceAttributes;
//     }
//     if(attr == EnumNameNodeAttributes(NodeAttributes_PointwiseAttributes))
//     {
//         return NodeAttributes_PointwiseAttributes;
//     }
//     if(attr == EnumNameNodeAttributes(NodeAttributes_BatchnormBackwardAttributes))
//     {
//         return NodeAttributes_BatchnormBackwardAttributes;
//     }
//     if(attr == EnumNameNodeAttributes(NodeAttributes_BatchnormAttributes))
//     {
//         return NodeAttributes_BatchnormAttributes;
//     }
//     if(attr == EnumNameNodeAttributes(NodeAttributes_ConvolutionFwdAttributes))
//     {
//         return NodeAttributes_ConvolutionFwdAttributes;
//     }
//     throw std::runtime_error(
//         fmt::format("Invalid NodeAttributes enum: {}", static_cast<int8_t>(attr)));
// }

NLOHMANN_JSON_SERIALIZE_ENUM(
    data_objects::NodeAttributes,
    {{data_objects::NodeAttributes_BatchnormInferenceAttributes, "BatchnormInferenceAttributes"},
     {data_objects::NodeAttributes_PointwiseAttributes, "PointwiseAttributes"},
     {data_objects::NodeAttributes_BatchnormBackwardAttributes, "BatchnormBackwardAttributes"},
     {data_objects::NodeAttributes_BatchnormAttributes, "BatchnormAttributes"},
     {data_objects::NodeAttributes_ConvolutionFwdAttributes, "ConvolutionFwdAttributes"},
     {data_objects::NodeAttributes_NONE, ""}})

NLOHMANN_JSON_SERIALIZE_ENUM(data_objects::DataType,
                             {
                                 {data_objects::DataType_UNSET, "unset"},
                                 {data_objects::DataType_FLOAT, "float"},
                                 {data_objects::DataType_HALF, "half"},
                                 {data_objects::DataType_BFLOAT16, "bfloat16"},
                                 {data_objects::DataType_DOUBLE, "double"},
                                 {data_objects::DataType_UINT8, "uint8"},
                                 {data_objects::DataType_INT32, "int32"},
                             }

)

nlohmann::json json(data_objects::Node const& node)
{
    auto type = node.attributes_type();
    nlohmann::json nodeJson = [&]() {
        if(type == data_objects::NodeAttributes_BatchnormInferenceAttributes)
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
    graphJson["node"] = nlohmann::json::array();

    for(auto node : *graph.nodes())
    {
        graphJson["node"].push_back(json(*node));
    }

    graphJson["compute_type"] = json(graph.compute_type());
    graphJson["io_type"] = json(graph.io_type());
    graphJson["intermediate_type"] = json(graph.intermediate_type());

    graphJson["name"] = graph.name()->c_str();

    // TODO: Handle tensors

    return graphJson;
}

/*
namespace details
{
template <class T>
struct IsConvertible;

template <>
struct IsConvertible<bool>
{
    bool operator()(const nlohmann::json& obj)
    {
        return obj.is_boolean();
    }
};

template <>
struct IsConvertible<std::string>
{
    bool operator()(const nlohmann::json& obj)
    {
        return obj.is_string();
    }
};

template <>
struct IsConvertible<float>
{
    bool operator()(const nlohmann::json& obj)
    {
        return obj.is_number();
    }
};

template <std::signed_integral T>
struct IsConvertible<T>
{
    bool operator()(const nlohmann::json& obj)
    {
        return obj.is_number_integer();
    }
};

template <std::unsigned_integral T>
struct IsConvertible<T>
{
    bool operator()(const nlohmann::json& obj)
    {
        return obj.is_number_unsigned();
    }
};

template <class T>
struct IsConvertible<std::vector<T>>
{
    bool operator()(const nlohmann::json& obj)
    {
        if(!obj.is_array())
        {
            return false;
        }

        return std::all_of(obj.begin(), obj.end(), IsConvertible<T>{});
    }
};

template <class T>
    requires(fmt::is_formattable<T>())
const T& formattableOrFallback(T const& v, const char*
{
    return v;
}

template <class T>
    requires(!fmt::is_formattable<T>())
const char* formattableOrFallback(T const&  const char* fallback)
{
    return fallback;
}

}

template <class T>
bool isConvertible(const nlohmann::json& obj)
{
    return IsConvertible<T>(obj);
}

template <class T, class Key>
T getWithDefault(const nlohmann::json& obj, Key&& key, T defaultVal)
{
    auto ret = obj.find(key);
    
    if(ret != obj.end())
    {
        return std::move(defaultVal);
    }
    
    if(!isConvertible<T>())
    {
        throw std::runtime_error(
            fmt::format("hipdnn_sdk::getWithDefault Key ({}) exists, but type is incompatible",
            details::formattableOrFallback(key, "<unformattable>")));
        }
        
        obj.value() return ret;
    }
*/

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
        case data_objects::NodeAttributes_BatchnormInferenceAttributes:
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
