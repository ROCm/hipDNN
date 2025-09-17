#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <nlohmann/json.hpp>

namespace hipdnn_sdk::json
{
auto fromGraph(void* flatbufferGraph)
{
    auto graph = data_objects::GetGraph(flatbufferGraph);
}
}
