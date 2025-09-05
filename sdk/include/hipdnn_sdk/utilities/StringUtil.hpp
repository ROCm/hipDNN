// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>

namespace hipdnn_sdk::utilities
{

static void copyMaxSizeWithNullTerminator(char* destination, const char* source, size_t maxSize)
{
    if(source == nullptr || destination == nullptr || maxSize == 0)
    {
        return;
    }

#ifdef _WIN32
    strncpy_s(destination, maxSize, source, maxSize - 1);
#else
    std::strncpy(destination, source, maxSize - 1);
#endif
    destination[maxSize - 1] = '\0';
}
}
