// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <memory>

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
        if(buffer)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineConfig>())
            {
                _engine_config
                    = flatbuffers::GetRoot<hipdnn_sdk::data_objects::EngineConfig>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::EngineConfig& get_engine_config() const override
    {
        return *_engine_config;
    }

    bool is_valid() const override
    {
        return _engine_config != nullptr;
    }

    int64_t engine_id() const override
    {
        return _engine_config ? _engine_config->engine_id() : -1;
    }

private:
    const hipdnn_sdk::data_objects::EngineConfig* _engine_config = nullptr;
};

}
