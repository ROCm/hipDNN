#include "hipdnn_sdk/plugin/plugin_api_data_types.h"
#include <hipdnn_sdk/plugin/engine_plugin_api.h>

extern "C" {
hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    *name = "test_incomplete_api_plugin";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    *version = "1.0.0";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}
}
