// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "handle/handle.hpp"
#include <gmock/gmock.h>

struct Mock_handle : hipdnnHandle
{
    MOCK_METHOD(void, set_stream, (hipStream_t stream), (override));
    MOCK_METHOD(hipStream_t, get_stream, (), (const, override));
    MOCK_METHOD(std::shared_ptr<hipdnn_backend::plugin::Engine_plugin_resource_manager>,
                get_plugin_resource_manager,
                (),
                (const, override));
};
