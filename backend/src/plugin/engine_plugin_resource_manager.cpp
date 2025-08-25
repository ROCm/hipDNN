// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <mutex>
#include <vector>

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "engine_plugin_manager.hpp"
#include "engine_plugin_resource_manager.hpp"
#include "hipdnn_exception.hpp"
#include "logging/logging.hpp"
#include <hipdnn_sdk/utilities/string_util.hpp>

namespace hipdnn_backend
{
namespace plugin
{

namespace
{

struct PluginLoadingConfig
{
    std::set<std::filesystem::path> paths;
    hipdnnPluginLoadingMode_ext_t mode = HIPDNN_DEFAULT_PLUGIN_LOADING_MODE;
};

std::mutex pluginMutex;
PluginLoadingConfig pluginConfig;
std::weak_ptr<EnginePluginManager> pmPtr;

} // namespace

void EnginePluginResourceManager::setPluginPaths(
    const std::vector<std::filesystem::path>& pluginPaths,
    hipdnnPluginLoadingMode_ext_t loadingMode)
{
    std::lock_guard<std::mutex> lock(pluginMutex);

    THROW_IF_FALSE(pmPtr.expired(),
                   HIPDNN_STATUS_NOT_SUPPORTED,
                   "hipdnnSetEnginePluginPaths_ext cannot be called with an active handle.");

    pluginConfig.mode = loadingMode;

    if(loadingMode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
    {
        pluginConfig.paths = {pluginPaths.begin(), pluginPaths.end()};
    }
    else
    {
        pluginConfig.paths.insert(pluginPaths.begin(), pluginPaths.end());
    }
}

std::set<std::filesystem::path> EnginePluginResourceManager::getPluginPaths()
{
    std::lock_guard<std::mutex> lock(pluginMutex);
    return pluginConfig.paths;
}

void EnginePluginResourceManager::getLoadedPluginFiles(size_t* numPlugins,
                                                       char** pluginPaths,
                                                       size_t* maxStringLen) const
{
    if(!_pm)
    {
        *numPlugins = 0;
        *maxStringLen = 0;
        return;
    }

    const auto& pathSet = _pm->getLoadedPluginFiles();

    size_t requiredLen = 0;
    for(const auto& path : pathSet)
    {
        requiredLen = std::max(requiredLen, path.string().length() + 1);
    }

    if(pluginPaths == nullptr)
    {
        *numPlugins = pathSet.size();
        *maxStringLen = requiredLen;
        return;
    }

    if(*numPlugins < pathSet.size() || *maxStringLen < requiredLen)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM, "Insufficient buffer space provided.");
    }

    std::vector<std::string> pathsVec;
    pathsVec.reserve(pathSet.size());
    pathsVec.assign(pathSet.begin(), pathSet.end());

    for(size_t i = 0; i < pathsVec.size(); ++i)
    {
        if(pluginPaths[i] == nullptr)
        {
            throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM, "A plugin path string buffer is null.");
        }
        hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
            pluginPaths[i], pathsVec[i].c_str(), *maxStringLen);
    }
}

std::shared_ptr<EnginePluginResourceManager> EnginePluginResourceManager::create()
{
    auto pm = pmPtr.lock();

    if(!pm)
    {
        std::lock_guard<std::mutex> lock(pluginMutex);

        pm = pmPtr.lock();

        if(!pm)
        {
            pm = std::make_shared<EnginePluginManager>();
            pm->loadPlugins(pluginConfig.paths, pluginConfig.mode);
            pmPtr = pm;
        }
    }

    return std::make_shared<EnginePluginResourceManager>(pm);
}

EnginePluginResourceManager::EnginePluginResourceManager()
    : _pm(std::make_shared<EnginePluginManager>())
{
}

EnginePluginResourceManager::EnginePluginResourceManager(std::shared_ptr<EnginePluginManager> pm)
    : _pm(std::move(pm))
{
    // Create plugin handles
    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        auto handle = plugin->createHandle();

        if(_handleToPlugin.contains(handle))
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Plugin handle already exists");
        }

        _handleToPlugin[handle] = plugin.get();

        auto engineIds = plugin->getAllEngineIds();
        for(const auto id : engineIds)
        {
            _engineIdToHandle[id] = handle;
        }
    }
}

EnginePluginResourceManager::~EnginePluginResourceManager()
{
    // Destroy plugin handles
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        try
        {
            plugin->destroyHandle(handle);
        }
        catch(const Hipdnn_exception& e)
        {
            HIPDNN_LOG_ERROR(e.get_message());
        }
    }
}

EnginePluginResourceManager::EnginePluginResourceManager(
    EnginePluginResourceManager&& other) noexcept
    : _pm(std::move(other._pm))
    , _handleToPlugin(std::move(other._handleToPlugin))
    , _engineIdToHandle(std::move(other._engineIdToHandle))
{
}

EnginePluginResourceManager&
    EnginePluginResourceManager::operator=(EnginePluginResourceManager&& other) noexcept
{
    if(this != &other)
    {
        _pm = std::move(other._pm);
        _handleToPlugin = std::move(other._handleToPlugin);
        _engineIdToHandle = std::move(other._engineIdToHandle);
    }
    return *this;
}

void EnginePluginResourceManager::setStream(hipStream_t stream) const
{
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        plugin->setStream(handle, stream);
    }
}

std::vector<int64_t>
    EnginePluginResourceManager::getApplicableEngineIds(const Graph_descriptor* graphDesc) const
{
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");

    auto serializedGraphData = graphDesc->get_serialized_graph();

    std::vector<int64_t> engineIds;

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        auto ids = plugin->getApplicableEngineIds(handle, &serializedGraphData);
        engineIds.insert(engineIds.end(), ids.begin(), ids.end());

        for(const auto& id : ids)
        {
            if(!_engineIdToHandle.contains(id))
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Unknown engine ID");
            }

            auto existingHandle = _engineIdToHandle.at(id);
            if(existingHandle != handle)
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                       "Engine ID " + std::to_string(id)
                                           + " is already associated with a different plugin");
            }
        }
    }

    return engineIds;
}

void EnginePluginResourceManager::getEngineDetails(int64_t engineId,
                                                   const Graph_descriptor* graphDesc,
                                                   hipdnnPluginConstData_t* engineDetails) const
{
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");
    THROW_IF_NULL(engineDetails, HIPDNN_STATUS_INTERNAL_ERROR, "Engine details cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->get_serialized_graph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    plugin->getEngineDetails(handle, engineId, &serializedGraphData, engineDetails);

    if(engineDetails->ptr == nullptr || engineDetails->size == 0)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Engine details for engine ID " + std::to_string(engineId)
                                   + " are empty or null");
    }
}

void EnginePluginResourceManager::destroyEngineDetails(int64_t engineId,
                                                       hipdnnPluginConstData_t* engineDetails) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->destroyEngineDetails(handle, engineDetails);
}

std::shared_ptr<const EngineDetailsWrapper> EnginePluginResourceManager::getEngineDetails(
    const std::shared_ptr<EnginePluginResourceManager>& rm,
    int64_t engineId,
    const Graph_descriptor* graphDesc)
{
    return std::make_shared<EngineDetailsWrapper>(rm, engineId, graphDesc);
}

size_t EnginePluginResourceManager::getWorkspaceSize(int64_t engineId,
                                                     const hipdnnPluginConstData_t* engineConfig,
                                                     const Graph_descriptor* graphDesc) const
{
    THROW_IF_NULL(engineConfig, HIPDNN_STATUS_INTERNAL_ERROR, "Engine config cannot be null");
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_INTERNAL_ERROR, "Graph descriptor cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->get_serialized_graph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->getWorkspaceSize(handle, engineConfig, &serializedGraphData);
}

// TODO: Pack engineConfig
// TODO: Get engineId from engineConfig
hipdnnEnginePluginExecutionContext_t
    EnginePluginResourceManager::createExecutionContext(int64_t engineId,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const Graph_descriptor* graphDesc) const
{
    THROW_IF_NULL(engineConfig, HIPDNN_STATUS_BAD_PARAM, "Engine config cannot be null");
    THROW_IF_NULL(graphDesc, HIPDNN_STATUS_BAD_PARAM, "Graph descriptor cannot be null");

    auto it = _engineIdToHandle.find(engineId);
    if(it == _engineIdToHandle.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Invalid engine ID: " + std::to_string(engineId));
    }

    auto serializedGraphData = graphDesc->get_serialized_graph();

    auto handle = it->second;
    auto plugin = _handleToPlugin.at(handle);

    return plugin->createExecutionContext(handle, engineConfig, &serializedGraphData);
}

void EnginePluginResourceManager::destroyExecutionContext(
    int64_t engineId, hipdnnEnginePluginExecutionContext_t executionContext) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->destroyExecutionContext(handle, executionContext);
}

std::shared_ptr<const EngineExecutionContextWrapper>
    EnginePluginResourceManager::createExecutionContext(
        const std::shared_ptr<EnginePluginResourceManager>& rm,
        int64_t engineId,
        const hipdnnPluginConstData_t* engineConfig,
        const Graph_descriptor* graphDesc)
{
    return std::make_shared<EngineExecutionContextWrapper>(rm, engineId, engineConfig, graphDesc);
}

void EnginePluginResourceManager::executeOpGraph(
    int64_t engineId,
    hipdnnEnginePluginExecutionContext_t executionContext,
    void* workspace,
    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
    uint32_t numDeviceBuffers) const
{
    auto handle = _engineIdToHandle.at(engineId);
    auto plugin = _handleToPlugin.at(handle);

    plugin->executeOpGraph(handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}

void EnginePluginResourceManager::executeOpGraph(hipdnnBackendDescriptor_t executionPlan,
                                                 hipdnnBackendDescriptor_t variantPack) const
{
    auto executionPlanDesc = executionPlan->as_descriptor<Execution_plan_descriptor>();
    auto variantPackDesc = variantPack->as_descriptor<Variant_descriptor>();

    THROW_IF_FALSE(executionPlanDesc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: executionPlanDesc "
                   "is not finalized");

    THROW_IF_FALSE(variantPackDesc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: variantPackDesc is "
                   "not finalized");

    auto config = executionPlanDesc->get_engine_config();
    auto engine = config->get_engine();
    auto engineId = engine->get_engine_id();
    void* workspace = variantPackDesc->get_workspace();

    auto& tensorIds = variantPackDesc->get_tensor_ids();
    auto& tensorPointers = variantPackDesc->get_data_pointers();

    THROW_IF_NE(tensorIds.size(),
                tensorPointers.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_plugin_resource_manager::execute_op_graph failed: "
                "tensorIds and tensorPointers must have the same size");

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;
    deviceBuffers.reserve(tensorIds.size());
    for(size_t i = 0; i < tensorIds.size(); ++i)
    {
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = tensorIds[i];
        buffer.ptr = const_cast<void*>(tensorPointers[i]);
        deviceBuffers.push_back(buffer);
    }

    executeOpGraph(engineId,
                   executionPlanDesc->get_execution_context(),
                   workspace,
                   deviceBuffers.data(),
                   static_cast<uint32_t>(tensorIds.size()));
}

EngineDetailsWrapper::EngineDetailsWrapper(const std::shared_ptr<EnginePluginResourceManager>& rm,
                                           int64_t engineId,
                                           const Graph_descriptor* graphDesc)
    : _rm(rm)
{
    _rm->getEngineDetails(engineId, graphDesc, &_engineDetailsData);
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(_engineDetailsData.ptr),
                                   _engineDetailsData.size);
    if(!verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineDetails>())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine_details_wrapper: unable to verify the flatbuffer schema.");
    }
}

EngineDetailsWrapper::~EngineDetailsWrapper()
{
    if(_engineDetailsData.ptr == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroyEngineDetails(get()->engine_id(), &_engineDetailsData);
    }
    catch(const Hipdnn_exception& e)
    {
        HIPDNN_LOG_ERROR(e.get_message());
    }
}

EngineDetailsWrapper::EngineDetailsWrapper(EngineDetailsWrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engineDetailsData(other._engineDetailsData)
{
    other._rm = nullptr;
    other._engineDetailsData.ptr = nullptr;
}

EngineDetailsWrapper& EngineDetailsWrapper::operator=(EngineDetailsWrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engineDetailsData = other._engineDetailsData;

        other._rm = nullptr;
        other._engineDetailsData.ptr = nullptr;
    }
    return *this;
}

const hipdnn_sdk::data_objects::EngineDetails* EngineDetailsWrapper::get() const
{
    if(_engineDetailsData.ptr == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "EngineDetailsWrapper: wrong usage: "
                               "get() called on an empty object");
    }

    return hipdnn_sdk::data_objects::GetEngineDetails(_engineDetailsData.ptr);
}

// TODO: Use engineId from engineConfig
EngineExecutionContextWrapper::EngineExecutionContextWrapper(
    const std::shared_ptr<EnginePluginResourceManager>& rm,
    int64_t engineId,
    const hipdnnPluginConstData_t* engineConfig,
    const Graph_descriptor* graphDesc)
    : _rm(rm)
    , _engineId(engineId)
{
    _executionContext = _rm->createExecutionContext(engineId, engineConfig, graphDesc);
}

EngineExecutionContextWrapper::~EngineExecutionContextWrapper()
{
    if(_executionContext == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroyExecutionContext(_engineId, _executionContext);
    }
    catch(const Hipdnn_exception& e)
    {
        HIPDNN_LOG_ERROR(e.get_message());
    }
}

EngineExecutionContextWrapper::EngineExecutionContextWrapper(
    EngineExecutionContextWrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engineId(other._engineId)
    , _executionContext(other._executionContext)
{
    other._rm = nullptr;
    other._executionContext = nullptr;
}

EngineExecutionContextWrapper&
    EngineExecutionContextWrapper::operator=(EngineExecutionContextWrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engineId = other._engineId;
        _executionContext = other._executionContext;

        other._rm = nullptr;
        other._executionContext = nullptr;
    }
    return *this;
}

hipdnnEnginePluginExecutionContext_t EngineExecutionContextWrapper::get() const
{
    if(_executionContext == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Engine_execution_context_wrapper: wrong usage: "
                               "get() called on an empty object");
    }

    return _executionContext;
}

} // namespace plugin
} // namespace hipdnn_backend
