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

static const char* pluginName = "miopen_legacy_plugin";
static const char* pluginVersion = "1.0.0";

using namespace hipdnn_plugin;
using namespace miopen_legacy_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

// Keep a weak pointer to the MiopenContainer thats made when we create a plugin handle.
// The original shared_ptr is then stored on the handle so that it can be used for the lifecycle
// of the handle.  If we create another handle, then we can use the weak pointer to get access
// to the existing MiopenContainer.  If all handles are destroyed, then this allows us to properly
// clean up the container without having to fully unload the plugin.
std::weak_ptr<MiopenContainer> miopenContainerLifecyclePtr;

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName={:p}", static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    LOG_API_ENTRY("versionPtr={:p}", static_cast<void*>(version));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version={:p}", static_cast<void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr={:p}", static_cast<void*>(type));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type={}", *type);
    });
}

void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(errorStr));

    hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(errorStr);

        *errorStr = Plugin_last_error_manager::get_last_error();

        LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(errorStr));
    });
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(callback);
        hipdnn::logging::initialize_callback_logging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        if(maxEngines != 0)
        {
            throw_if_null(engineIds);
        }
        throw_if_null(numEngines);

        // For now, we will just return a single engine ID.
        auto allEngineIds = std::vector<int64_t>({1});
        if(allEngineIds.size() > std::numeric_limits<uint32_t>::max())
        {
            throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                          "Number of engines exceeds maximum uint32_t value.");
        }

        if(maxEngines == 0)
        {
            *numEngines = static_cast<uint32_t>(allEngineIds.size());
        }
        else
        {
            *numEngines = 0;
            for(auto engineId : allEngineIds)
            {
                if(*numEngines == maxEngines)
                {
                    *numEngines = static_cast<uint32_t>(allEngineIds.size());
                    HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                                    "engines, numEngines count: {}",
                                    maxEngines,
                                    *numEngines);
                    break;
                }

                engineIds[*numEngines] = engineId;
                (*numEngines)++;
            }
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);

        miopen_legacy_plugin::MiopenHandleFactory::createMiopenHandle(handle);

        auto miopenContainerPtr = miopenContainerLifecyclePtr.lock();
        if(miopenContainerPtr != nullptr)
        {
            (*handle)->miopenContainer = miopenContainerPtr;
        }
        else
        {
            static std::mutex miopenContainerMutex;
            std::lock_guard<std::mutex> lock(miopenContainerMutex);

            // if we do have a race condition that results in threads getting locked, we want to
            // ensure that we only create one instance.  Therefore, the second thread to get
            // through will just read from the weak pointer rather than create a new instance.
            miopenContainerPtr = miopenContainerLifecyclePtr.lock();
            if(miopenContainerPtr != nullptr)
            {
                (*handle)->miopenContainer = miopenContainerPtr;
            }
            else
            {
                (*handle)->miopenContainer = std::make_shared<MiopenContainer>();
                miopenContainerLifecyclePtr = (*handle)->miopenContainer;
            }
        }

        LOG_API_SUCCESS(apiName, "createdHandle={:p}", static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);

        miopen_legacy_plugin::MiopenHandleFactory::destroyMiopenHandle(handle);
        delete handle;
        handle = nullptr;

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    LOG_API_ENTRY(
        "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);

        handle->setStream(stream);

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(opGraph);
        if(maxEngines != 0)
        {
            throw_if_null(engineIds);
        }
        throw_if_null(numEngines);

        auto& engineManager = handle->getEngineManager();
        Graph_wrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        auto applicableEngines = engineManager.getApplicableEngineIds(opGraphWrapper);

        *numEngines = 0;
        for(auto& engineId : applicableEngines)
        {
            if(*numEngines == maxEngines)
            {
                *numEngines = static_cast<uint32_t>(applicableEngines.size());
                HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                                "engines, numEngines count: {}",
                                maxEngines,
                                *numEngines);
                break;
            }

            engineIds[*numEngines] = engineId;
            (*numEngines)++;
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engineId,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                  static_cast<void*>(handle),
                  engineId,
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(opGraph);
        throw_if_null(engineDetails);

        auto& engineManager = handle->getEngineManager();
        Graph_wrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        engineManager.getEngineDetails(*handle, opGraphWrapper, engineId, *engineDetails);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineDetails={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(engineDetails);
        throw_if_null(engineDetails->ptr);

        handle->removeEngineDetailsDetachedBuffer(engineDetails->ptr);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        size_t* workspaceSize)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(workspaceSize));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(engineConfig);
        throw_if_null(opGraph);
        throw_if_null(workspaceSize);

        auto& engineManager = handle->getEngineManager();

        Engine_config_wrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);
        Graph_wrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        *workspaceSize = engineManager.getWorkspaceSize(
            *handle, engineConfigWrapper.engine_id(), opGraphWrapper);

        LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(engineConfig);
        throw_if_null(opGraph);
        throw_if_null(executionContext);

        Graph_wrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        Engine_config_wrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);

        auto& engineManager = handle->getEngineManager();

        auto context = new HipdnnEnginePluginExecutionContext;

        try
        {
            engineManager.initializeExecutionContext(
                *handle, opGraphWrapper, engineConfigWrapper, *context);
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *executionContext = context;

        LOG_API_SUCCESS(
            apiName, "created_execution_context={:p}", static_cast<void*>(*executionContext));
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(executionContext);

        delete executionContext;

        LOG_API_SUCCESS(apiName, "destroyed executionContext");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, "
                  "numDeviceBuffers={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext),
                  workspace,
                  static_cast<const void*>(deviceBuffers),
                  numDeviceBuffers);

    return hipdnn_plugin::try_catch([&, apiName = __func__]() {
        throw_if_null(handle);
        throw_if_null(executionContext);
        throw_if_null(deviceBuffers);

        executionContext->plan().execute(*handle, deviceBuffers, numDeviceBuffers, workspace);

        LOG_API_SUCCESS(apiName, "executed graph");
    });
}

} // extern "C"
