// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle.hpp"

void hipdnnHandle::set_stream(hipStream_t stream)
{
    _stream = stream;
}

hipStream_t hipdnnHandle::get_stream() const
{
    return _stream;
}
