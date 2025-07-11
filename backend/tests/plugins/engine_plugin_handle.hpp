// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.

#pragma once

#include <hip/hip_runtime.h>

struct hipdnnEnginePluginHandle
{
    hipStream_t stream;
};
