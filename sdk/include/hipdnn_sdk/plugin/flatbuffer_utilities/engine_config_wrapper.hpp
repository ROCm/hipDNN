// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

namespace hipdnn_plugin
{

class Engine_config_interface
{
public:
    virtual ~Engine_config_interface() = default;

    virtual const hipdnn_sdk::data_objects::EngineConfig& get_engine_config() const = 0;
    virtual bool is_valid() const = 0;
    virtual int64_t engine_id() const = 0;
};

class Engine_config_wrapper : public Engine_config_interface
{
public:
    explicit Engine_config_wrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineConfig>())
            {
                _shallow_engine_config
                    = flatbuffers::GetRoot<hipdnn_sdk::data_objects::EngineConfig>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::EngineConfig& get_engine_config() const override
    {
        throw_if_not_valid();

        return *_shallow_engine_config;
    }

    bool is_valid() const override
    {
        return _shallow_engine_config != nullptr;
    }

    int64_t engine_id() const override
    {
        throw_if_not_valid();

        return _shallow_engine_config->engine_id();
    }

private:
    void throw_if_not_valid() const
    {
        if(!is_valid())
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         "Engine config is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the engine config. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::EngineConfig* _shallow_engine_config = nullptr;
};

}
