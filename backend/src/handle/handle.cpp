// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle.hpp"

namespace hipdnn_backend
{
void Handle::set_stream(hipStream_t stream)
{
    _stream = stream;
}

hipStream_t Handle::get_stream() const
{
    return _stream;
}

}