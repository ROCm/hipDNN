#include "hipdnn_backend.h"
#include <iostream>

#include "hello_world.hpp"
#include <string.h>

int publicFunctionHello()
{
    std::cout << hipdnn_backend::HelloWorld::getMessage() << std::endl;

    char buffer[10];
    strcpy(buffer, "This string is too long for the buffer");
    return 1337;
}