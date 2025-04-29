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
};

namespace hipdnn_backend
{
class Handle : public hipdnnHandle
{
public:
    ~Handle() override = default;
    void        set_stream(hipStream_t stream);
    hipStream_t get_stream() const;
    hipStream_t _stream = nullptr;
};
}