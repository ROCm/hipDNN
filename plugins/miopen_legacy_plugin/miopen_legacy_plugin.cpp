// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <miopen/miopen.h>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_config_wrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include "engine_manager.hpp"
#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"
#include "miopen_container.hpp"
#include "miopen_handle_factory.hpp"

static const char* _plugin_name = "miopen_legacy_plugin";
static const char* _plugin_version = "1.0.0";

using namespace hipdnn_plugin;
using namespace miopen_legacy_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__ __VA_OPT__(, ) __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name __VA_OPT__(, ) __VA_ARGS__)

template <typename T>
void throw_if_null(T* value)
{
    if(value == nullptr)
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                     std::string(typeid(T).name()) + " is nullptr");
    }
}

// Keep a weak pointer to the Miopen_container thats made when we create a plugin handle.
// The original shared_ptr is then stored on the handle so that it can be used for the lifecycle
// of the handle.  If we create another handle, then we can use the weak pointer to get access
// to the existing Miopen_container.  If all handles are destroyed, then this allows us to properly
// clean up the container without having to fully unload the plugin.
std::weak_ptr<Miopen_container> miopen_container_lifecycle_ptr;

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(name);

        *name = _plugin_name;

        LOG_API_SUCCESS(api_name, "plugin_name={:p}", static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    LOG_API_ENTRY("version_ptr={:p}", static_cast<void*>(version));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(version);

        *version = _plugin_version;

        LOG_API_SUCCESS(api_name, "version={:p}", static_cast<void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("type_ptr={:p}", static_cast<void*>(type));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(api_name, "type={}", *type);
    });
}

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    LOG_API_ENTRY("error_str_ptr={:p}", static_cast<void*>(error_str));

    hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(error_str);

        *error_str = Plugin_last_error_manager::get_last_error();

        LOG_API_SUCCESS(api_name, "error_str={:p}", static_cast<void*>(error_str));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        miopen_legacy_plugin::Miopen_handle_factory::create_miopen_handle(handle);

        auto miopen_container_ptr = miopen_container_lifecycle_ptr.lock();
        if(miopen_container_ptr != nullptr)
        {
            (*handle)->miopen_container = miopen_container_ptr;
        }
        else
        {
            static std::mutex miopen_container_mutex;
            std::lock_guard<std::mutex> lock(miopen_container_mutex);

            // if we do have a race condition that results in threads getting locked, we want to
            // ensure that we only create one instance.  Therefore, the second thread to get
            // through will just read from the weak pointer rather than create a new instance.
            miopen_container_ptr = miopen_container_lifecycle_ptr.lock();
            if(miopen_container_ptr != nullptr)
            {
                (*handle)->miopen_container = miopen_container_ptr;
            }
            else
            {
                (*handle)->miopen_container = std::make_shared<Miopen_container>();
                miopen_container_lifecycle_ptr = (*handle)->miopen_container;
            }
        }

        LOG_API_SUCCESS(api_name, "created_handle={:p}", static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        miopen_legacy_plugin::Miopen_handle_factory::destroy_miopen_handle(handle);
        delete handle;
        handle = nullptr;

        LOG_API_SUCCESS(api_name, "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    LOG_API_ENTRY(
        "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        handle->stream = stream;

        LOG_API_SUCCESS(api_name, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph,
                                             int64_t* engine_ids,
                                             uint32_t max_engines,
                                             uint32_t* num_engines)
{
    LOG_API_ENTRY("handle={:p}, op_graph={:p}, engine_ids={:p}, max_engines={}, num_engines={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(op_graph),
                  static_cast<void*>(engine_ids),
                  max_engines,
                  static_cast<void*>(num_engines));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(op_graph);
        throw_if_null(engine_ids);
        throw_if_null(num_engines);

        auto& engine_manager = handle->get_engine_manager();
        Graph_wrapper op_graph_wrapper(op_graph->ptr, op_graph->size);

        auto applicable_engines = engine_manager.get_applicable_engine_ids(op_graph_wrapper);

        *num_engines = 0;
        for(auto& engine_id : applicable_engines)
        {
            if(*num_engines == max_engines)
            {
                *num_engines = static_cast<uint32_t>(applicable_engines.size());
                HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                                "engines, num_engines count: {}",
                                max_engines,
                                *num_engines);
                break;
            }

            engine_ids[*num_engines] = engine_id;
            (*num_engines)++;
        }

        LOG_API_SUCCESS(api_name, "num_engines={}", *num_engines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engine_id,
                                                        const hipdnnPluginConstData_t* op_graph,
                                                        hipdnnPluginConstData_t* engine_details)
{
    LOG_API_ENTRY("handle={:p}, engine_id={}, op_graph={:p}, engine_details={:p}",
                  static_cast<void*>(handle),
                  engine_id,
                  static_cast<const void*>(op_graph),
                  static_cast<void*>(engine_details));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(op_graph);
        throw_if_null(engine_details);

        auto& engine_manager = handle->get_engine_manager();
        Graph_wrapper op_graph_wrapper(op_graph->ptr, op_graph->size);

        engine_manager.get_engine_details(op_graph_wrapper, engine_id, *engine_details);

        LOG_API_SUCCESS(api_name, "engine_details->ptr={:p}", engine_details->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engine_details)
{
    LOG_API_ENTRY("handle={:p}, engine_details={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(engine_details));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(engine_details);
        throw_if_null(engine_details->ptr);

        delete[] static_cast<const uint8_t*>(engine_details->ptr);

        LOG_API_SUCCESS(api_name, "engine_details->ptr={:p}", engine_details->ptr);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size)
{
    LOG_API_ENTRY("handle={:p}, engine_config={:p}, op_graph={:p}, workspace_size={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engine_config),
                  static_cast<const void*>(op_graph),
                  static_cast<void*>(workspace_size));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(engine_config);
        throw_if_null(op_graph);
        throw_if_null(workspace_size);

        auto& engine_manager = handle->get_engine_manager();

        Engine_config_wrapper engine_config_wrapper(engine_config->ptr, engine_config->size);
        Graph_wrapper op_graph_wrapper(op_graph->ptr, op_graph->size);
        *workspace_size = engine_manager.get_workspace_size(
            *handle, engine_config_wrapper.engine_id(), op_graph_wrapper);

        LOG_API_SUCCESS(api_name, "workspace_size={}", *workspace_size);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engine_config,
    const hipdnnPluginConstData_t* op_graph,
    hipdnnEnginePluginExecutionContext_t* execution_context)
{
    LOG_API_ENTRY("handle={:p}, engine_config={:p}, op_graph={:p}, execution_context={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engine_config),
                  static_cast<const void*>(op_graph),
                  static_cast<void*>(execution_context));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(engine_config);
        throw_if_null(op_graph);
        throw_if_null(execution_context);

        Graph_wrapper op_graph_wrapper(op_graph->ptr, op_graph->size);
        Engine_config_wrapper engine_config_wrapper(engine_config->ptr, engine_config->size);

        auto& engine_manager = handle->get_engine_manager();

        auto context = new hipdnnEnginePluginExecutionContext;

        try
        {
            engine_manager.initialize_execution_context(
                *handle, op_graph_wrapper, engine_config_wrapper, *context);
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *execution_context = context;

        LOG_API_SUCCESS(
            api_name, "created_execution_context={:p}", static_cast<void*>(*execution_context));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context)
{
    LOG_API_ENTRY("handle={:p}, execution_context={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(execution_context));

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(execution_context);

        delete execution_context;

        LOG_API_SUCCESS(api_name, "destroyed execution_context");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers)
{
    LOG_API_ENTRY("handle={:p}, execution_context={:p}, workspace={:p}, device_buffers={:p}, "
                  "num_device_buffers={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(execution_context),
                  workspace,
                  static_cast<const void*>(device_buffers),
                  num_device_buffers);

    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(execution_context);
        throw_if_null(device_buffers);

        execution_context->plan().execute(*handle, device_buffers, num_device_buffers, workspace);

        LOG_API_SUCCESS(api_name, "executed graph");
    });
}

} // extern "C"
