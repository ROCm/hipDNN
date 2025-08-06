// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "formatting.hpp"
#include <spdlog/details/log_msg.h>
#include <spdlog/pattern_formatter.h>
#include <string>
#include <vector>

namespace hipdnn::logging
{

class Component_formatter final : public spdlog::formatter
{
public:
    Component_formatter()
        : _pass_through_formatter{std::make_unique<spdlog::pattern_formatter>("%v")}
    {
    }

    void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override
    {
        // The logger "hipdnn_callback_receiver" receives pre-formatted strings from a callback sink
        if(msg.logger_name == "hipdnn_callback_receiver")
        {
            _pass_through_formatter->format(msg, dest);
        }
        else
        {
            auto standard_formatter
                = std::make_unique<spdlog::pattern_formatter>(generate_pattern_string(
                    std::string(msg.logger_name.data(), msg.logger_name.size())));
            standard_formatter->format(msg, dest);
        }
    }

    std::unique_ptr<spdlog::formatter> clone() const override
    {
        return std::make_unique<Component_formatter>();
    }

private:
    std::unique_ptr<spdlog::pattern_formatter> _pass_through_formatter;
};

} // namespace hipdnn::logging
