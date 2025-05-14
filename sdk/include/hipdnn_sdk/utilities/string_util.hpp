// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>

namespace hipdnn::sdk::utilities
{

    static void copy_max_size_with_null_terminator(char* destination, const char* source, size_t max_size)
    {
        if(!source || !destination || max_size == 0)
        {
            return;
        }

        std::strncpy(destination, source, max_size - 1);
        destination[max_size - 1] = '\0';
    }
}
