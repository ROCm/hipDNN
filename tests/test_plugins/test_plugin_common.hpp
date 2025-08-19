// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <iostream>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_config_wrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

struct hipdnnEnginePluginHandle
{
public:
    virtual ~hipdnnEnginePluginHandle() = default;
};

struct hipdnnEnginePluginExecutionContext
{
};

// Base class for test plugins
class Test_plugin_base
{
public:
    virtual ~Test_plugin_base() = default;

    // Virtual methods to be overridden by derived classes
    virtual const char* get_plugin_name() const = 0;
    virtual const char* get_plugin_version() const = 0;
    virtual int64_t get_engine_id() const = 0;
    virtual uint32_t get_num_engines() const = 0;
    virtual uint32_t get_num_applicable_engines() const = 0;
    virtual bool supports_engine_operations() const
    {
        return get_num_applicable_engines() > 0;
    }

    // Execute graph - derived classes override this for custom behavior
    virtual void execute_graph() const
    {
        HIPDNN_LOG_INFO("execute_graph called");
    }

    // Static instance management
    static void set_instance(std::unique_ptr<Test_plugin_base> instance)
    {
        _instance = std::move(instance);
    }

    static Test_plugin_base* get_instance()
    {
        return _instance.get();
    }

    // Common API implementations
    static hipdnnPluginStatus_t plugin_get_name(const char** name)
    {
        LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(name);
            hipdnn_plugin::throw_if_null(get_instance());

            *name = get_instance()->get_plugin_name();

            LOG_API_SUCCESS(api_name, "plugin_name={:p}", static_cast<void*>(name));
        });
    }

    static hipdnnPluginStatus_t plugin_get_version(const char** version)
    {
        LOG_API_ENTRY("version_ptr={:p}", static_cast<void*>(version));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(version);
            hipdnn_plugin::throw_if_null(get_instance());

            *version = get_instance()->get_plugin_version();

            LOG_API_SUCCESS(api_name, "version={:p}", static_cast<void*>(version));
        });
    }

    static hipdnnPluginStatus_t plugin_get_type(hipdnnPluginType_t* type)
    {
        LOG_API_ENTRY("type_ptr={:p}", static_cast<void*>(type));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(type);

            *type = HIPDNN_PLUGIN_TYPE_ENGINE;

            LOG_API_SUCCESS(api_name, "type={}", *type);
        });
    }

    static void plugin_get_last_error_string(const char** error_str)
    {
        LOG_API_ENTRY("error_str_ptr={:p}", static_cast<void*>(error_str));

        hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(error_str);

            *error_str = hipdnn_plugin::Plugin_last_error_manager::get_last_error();

            LOG_API_SUCCESS(api_name, "error_str={:p}", static_cast<void*>(error_str));
        });
    }

    static hipdnnPluginStatus_t plugin_set_logging_callback(hipdnnCallback_t callback)
    {
        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(callback);
            hipdnn_plugin::throw_if_null(get_instance());

            hipdnn::logging::initialize_callback_logging(get_instance()->get_plugin_name(),
                                                         callback);
            LOG_API_SUCCESS(api_name, "");
        });
    }

    static hipdnnPluginStatus_t engine_plugin_get_all_engine_ids(int64_t* engine_ids,
                                                                 uint32_t max_engines,
                                                                 uint32_t* num_engines)
    {
        LOG_API_ENTRY("engine_ids={:p}, max_engines={}, num_engines={:p}",
                      static_cast<void*>(engine_ids),
                      max_engines,
                      static_cast<void*>(num_engines));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            if(max_engines != 0)
            {
                hipdnn_plugin::throw_if_null(engine_ids);
            }
            hipdnn_plugin::throw_if_null(num_engines);
            hipdnn_plugin::throw_if_null(get_instance());

            *num_engines = get_instance()->get_num_engines();

            if(max_engines >= 1 && *num_engines > 0)
            {
                assert(*num_engines == 1);
                engine_ids[0] = get_instance()->get_engine_id();
            }

            LOG_API_SUCCESS(api_name, "num_engines={}", *num_engines);
        });
    }

    static hipdnnPluginStatus_t engine_plugin_create(hipdnnEnginePluginHandle_t* handle)
    {
        LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);

            *handle = new hipdnnEnginePluginHandle();

            LOG_API_SUCCESS(api_name, "created_handle={:p}", static_cast<void*>(*handle));
        });
    }

    static hipdnnPluginStatus_t engine_plugin_destroy(hipdnnEnginePluginHandle_t handle)
    {
        LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);

            delete handle;
            handle = nullptr;

            LOG_API_SUCCESS(api_name, "");
        });
    }

    static hipdnnPluginStatus_t engine_plugin_set_stream(hipdnnEnginePluginHandle_t handle,
                                                         hipStream_t stream)
    {
        LOG_API_ENTRY(
            "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);

            LOG_API_SUCCESS(api_name, "");
        });
    }

    static hipdnnPluginStatus_t
        engine_plugin_get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                                                const hipdnnPluginConstData_t* op_graph,
                                                int64_t* engine_ids,
                                                uint32_t max_engines,
                                                uint32_t* num_engines)
    {
        LOG_API_ENTRY(
            "handle={:p}, op_graph={:p}, engine_ids={:p}, max_engines={}, num_engines={:p}",
            static_cast<void*>(handle),
            static_cast<const void*>(op_graph),
            static_cast<void*>(engine_ids),
            max_engines,
            static_cast<void*>(num_engines));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(op_graph);
            if(max_engines != 0)
            {
                hipdnn_plugin::throw_if_null(engine_ids);
            }
            hipdnn_plugin::throw_if_null(num_engines);
            hipdnn_plugin::throw_if_null(get_instance());

            *num_engines = get_instance()->get_num_applicable_engines();

            if(max_engines >= 1 && *num_engines > 0)
            {
                engine_ids[0] = get_instance()->get_engine_id();
            }

            LOG_API_SUCCESS(api_name, "num_engines={}", *num_engines);
        });
    }

    static hipdnnPluginStatus_t
        engine_plugin_get_engine_details(hipdnnEnginePluginHandle_t handle,
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
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(op_graph);
            hipdnn_plugin::throw_if_null(engine_details);
            hipdnn_plugin::throw_if_null(get_instance());

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get engine details");
            }

            flatbuffers::FlatBufferBuilder builder;
            auto new_engine_details = hipdnn_sdk::data_objects::CreateEngineDetails(
                builder, get_instance()->get_engine_id());
            builder.Finish(new_engine_details);
            auto serialized_details = builder.Release();

            auto* temp_buffer = new uint8_t[serialized_details.size()];
            std::memcpy(temp_buffer, serialized_details.data(), serialized_details.size());

            engine_details->ptr = temp_buffer;
            engine_details->size = serialized_details.size();

            LOG_API_SUCCESS(api_name, "engine_details->ptr={:p}", engine_details->ptr);
        });
    }

    static hipdnnPluginStatus_t
        engine_plugin_destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                                             hipdnnPluginConstData_t* engine_details)
    {
        LOG_API_ENTRY("handle={:p}, engine_details={}",
                      static_cast<void*>(handle),
                      static_cast<void*>(engine_details));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(engine_details);

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                             "No engine details to destroy");
            }

            hipdnn_plugin::throw_if_null(engine_details->ptr);

            delete[] static_cast<const uint8_t*>(engine_details->ptr);

            LOG_API_SUCCESS(api_name, "engine_details->ptr={:p}", engine_details->ptr);
        });
    }

    static hipdnnPluginStatus_t
        engine_plugin_get_workspace_size(hipdnnEnginePluginHandle_t handle,
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
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(engine_config);
            hipdnn_plugin::throw_if_null(op_graph);
            hipdnn_plugin::throw_if_null(workspace_size);
            hipdnn_plugin::throw_if_null(get_instance());

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get workspace size");
            }

            *workspace_size = 1024;

            LOG_API_SUCCESS(api_name, "workspace_size={}", *workspace_size);
        });
    }

    static hipdnnPluginStatus_t engine_plugin_create_execution_context(
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
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(engine_config);
            hipdnn_plugin::throw_if_null(op_graph);
            hipdnn_plugin::throw_if_null(execution_context);
            hipdnn_plugin::throw_if_null(get_instance());

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot create execution context");
            }

            hipdnn_plugin::Graph_wrapper op_graph_wrapper(op_graph->ptr, op_graph->size);
            hipdnn_plugin::Engine_config_wrapper engine_config_wrapper(engine_config->ptr,
                                                                       engine_config->size);

            *execution_context = new hipdnnEnginePluginExecutionContext();

            LOG_API_SUCCESS(
                api_name, "created_execution_context={:p}", static_cast<void*>(*execution_context));
        });
    }

    static hipdnnPluginStatus_t engine_plugin_destroy_execution_context(
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context)
    {
        LOG_API_ENTRY("handle={:p}, execution_context={:p}",
                      static_cast<void*>(handle),
                      static_cast<void*>(execution_context));

        return hipdnn_plugin::try_catch([&, api_name = __func__]() {
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(execution_context);
            hipdnn_plugin::throw_if_null(get_instance());

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                             "No execution context to destroy");
            }

            delete execution_context;

            LOG_API_SUCCESS(api_name, "destroyed execution_context");
        });
    }

    static hipdnnPluginStatus_t
        engine_plugin_execute_op_graph(hipdnnEnginePluginHandle_t handle,
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
            hipdnn_plugin::throw_if_null(handle);
            hipdnn_plugin::throw_if_null(execution_context);
            hipdnn_plugin::throw_if_null(device_buffers);
            hipdnn_plugin::throw_if_null(get_instance());

            if(!get_instance()->supports_engine_operations())
            {
                throw hipdnn_plugin::Hipdnn_plugin_exception(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot execute graph");
            }

            get_instance()->execute_graph();

            LOG_API_SUCCESS(api_name, "executed graph");
        });
    }

private:
    inline static std::unique_ptr<Test_plugin_base> _instance;
};

// Macro to register plugin API functions
#define REGISTER_TEST_PLUGIN_API()                                                                 \
    extern "C" {                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                    \
    {                                                                                              \
        return Test_plugin_base::plugin_get_name(name);                                            \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                              \
    {                                                                                              \
        return Test_plugin_base::plugin_get_version(version);                                      \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                             \
    {                                                                                              \
        return Test_plugin_base::plugin_get_type(type);                                            \
    }                                                                                              \
                                                                                                   \
    void hipdnnPluginGetLastErrorString(const char** error_str)                                    \
    {                                                                                              \
        Test_plugin_base::plugin_get_last_error_string(error_str);                                 \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                 \
    {                                                                                              \
        return Test_plugin_base::plugin_set_logging_callback(callback);                            \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t* engine_ids,                    \
                                                           uint32_t max_engines,                   \
                                                           uint32_t* num_engines)                  \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_get_all_engine_ids(                                 \
            engine_ids, max_engines, num_engines);                                                 \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)              \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_create(handle);                                     \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)              \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_destroy(handle);                                    \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,            \
                                                     hipStream_t stream)                           \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_set_stream(handle, stream);                         \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t                                                                           \
        hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,                \
                                                 const hipdnnPluginConstData_t* op_graph,          \
                                                 int64_t* engine_ids,                              \
                                                 uint32_t max_engines,                             \
                                                 uint32_t* num_engines)                            \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_get_applicable_engine_ids(                          \
            handle, op_graph, engine_ids, max_engines, num_engines);                               \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t                                                                           \
        hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,                      \
                                           int64_t engine_id,                                      \
                                           const hipdnnPluginConstData_t* op_graph,                \
                                           hipdnnPluginConstData_t* engine_details)                \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_get_engine_details(                                 \
            handle, engine_id, op_graph, engine_details);                                          \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t                                                                           \
        hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,                  \
                                               hipdnnPluginConstData_t* engine_details)            \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_destroy_engine_details(handle, engine_details);     \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t                                                                           \
        hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,                      \
                                           const hipdnnPluginConstData_t* engine_config,           \
                                           const hipdnnPluginConstData_t* op_graph,                \
                                           size_t* workspace_size)                                 \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_get_workspace_size(                                 \
            handle, engine_config, op_graph, workspace_size);                                      \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(                                 \
        hipdnnEnginePluginHandle_t handle,                                                         \
        const hipdnnPluginConstData_t* engine_config,                                              \
        const hipdnnPluginConstData_t* op_graph,                                                   \
        hipdnnEnginePluginExecutionContext_t* execution_context)                                   \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_create_execution_context(                           \
            handle, engine_config, op_graph, execution_context);                                   \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(                                \
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context) \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_destroy_execution_context(handle,                   \
                                                                         execution_context);       \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t                                                                           \
        hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,                        \
                                         hipdnnEnginePluginExecutionContext_t execution_context,   \
                                         void* workspace,                                          \
                                         const hipdnnPluginDeviceBuffer_t* device_buffers,         \
                                         uint32_t num_device_buffers)                              \
    {                                                                                              \
        return Test_plugin_base::engine_plugin_execute_op_graph(                                   \
            handle, execution_context, workspace, device_buffers, num_device_buffers);             \
    }                                                                                              \
    } // extern "C"
