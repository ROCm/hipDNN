#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <nlohmann/json.hpp>

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

}
