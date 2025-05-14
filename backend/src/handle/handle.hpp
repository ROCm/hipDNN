// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>

struct hipdnnHandle // NOLINT
{
public:
    virtual ~hipdnnHandle() = default;
    void set_stream(hipStream_t stream);
    hipStream_t get_stream() const;
    hipStream_t _stream = nullptr;
};
