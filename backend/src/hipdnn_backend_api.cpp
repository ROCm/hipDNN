// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <iostream>

#include "hello_world.hpp"

int publicFunctionHello()
{
    std::cout << hipdnn_backend::HelloWorld::getMessage() << std::endl;
    return 1337;
}