// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

namespace hipdnn_plugin
{

class Engine_details_interface
{
public:
    virtual ~Engine_details_interface() = default;

    virtual const hipdnn_sdk::data_objects::EngineDetails& get_engine_details() const = 0;
    virtual bool is_valid() const = 0;
    virtual int64_t engine_id() const = 0;
};

class Engine_details_wrapper : public Engine_details_interface
{
public:
    explicit Engine_details_wrapper(const void* buffer, size_t size)
    {
        if(buffer != nullptr)
        {
            flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
            if(verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineDetails>())
            {
                _shallow_engine_details
                    = flatbuffers::GetRoot<hipdnn_sdk::data_objects::EngineDetails>(buffer);
            }
        }
    }

    const hipdnn_sdk::data_objects::EngineDetails& get_engine_details() const override
    {
        throw_if_not_valid();
        return *_shallow_engine_details;
    }

    bool is_valid() const override
    {
        return _shallow_engine_details != nullptr;
    }

    int64_t engine_id() const override
    {
        throw_if_not_valid();

        return _shallow_engine_details->engine_id();
    }

private:
    void throw_if_not_valid() const
    {
        if(!is_valid())
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                         "Engine details is not valid");
        }
    }

    // Pointer to the flatbuffer representation of the engine details. We do not own this memory
    // as were just reading from the buffer passed during construction.
    const hipdnn_sdk::data_objects::EngineDetails* _shallow_engine_details = nullptr;
};

}
