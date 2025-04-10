// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <iostream>

#include "hello_world.hpp"

int public_function_hello()
{
    std::cout << hipdnn_backend::Hello_world::get_message();
    return 1337;
}