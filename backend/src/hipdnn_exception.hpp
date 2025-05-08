// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_status.h"
#include <exception>
#include <string>

namespace hipdnn_backend
{

class Hipdnn_exception : public std::exception
{
public:
    explicit Hipdnn_exception(hipdnnStatus_t status, std::string message)
        : _status(status)
        , _message(std::move(message))
    {
    }

    const char* what() const noexcept override
    {
        return _message.c_str();
    }

    std::string get_message() const noexcept
    {
        return _message;
    }

    hipdnnStatus_t get_status() const noexcept
    {
        return _status;
    }

private:
    hipdnnStatus_t _status;
    std::string _message;
};

} // namespace hipdnn_backend
