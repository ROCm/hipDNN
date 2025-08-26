// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "handle/handle.hpp"
#include <gmock/gmock.h>

struct Mock_handle : hipdnnHandle
{
    MOCK_METHOD(void, setStream, (hipStream_t stream), (override));
    MOCK_METHOD(hipStream_t, getStream, (), (const, override));
    MOCK_METHOD(std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager>,
                getPluginResourceManager,
                (),
                (const, override));
};
