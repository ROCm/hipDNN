// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "callback_types.h"
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/synchronous_factory.h>
#include <spdlog/sinks/base_sink.h>
#include <functional>
#include <mutex>
#include <string>

namespace hipdnn::logging
{


inline hipdnnSeverity_t spdlog_to_hipdnn_severity(spdlog::level::level_enum level)
{
    switch(level)
    {
    case spdlog::level::critical:
        return HIPDNN_SEV_FATAL;
    case spdlog::level::err:
        return HIPDNN_SEV_ERROR;
    case spdlog::level::warn:
        return HIPDNN_SEV_WARNING;
    case spdlog::level::info:
        return HIPDNN_SEV_INFO;
    default:
        return HIPDNN_SEV_INFO;
    }
}


template <typename Mutex>
class Callback_sink final : public spdlog::sinks::base_sink<Mutex>
{
public:
    explicit Callback_sink(hipdnnCallback_t callback, void* user_data, const std::string& source)
        : _callback_fn{callback}
        , _udata{user_data}
        , _source{source.c_str()} // accept std::string
    {
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        if (_callback_fn == nullptr)
        {
            return;
        }

        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string formatted_str(formatted.data(), formatted.size());

        hipdnnSeverity_t severity = spdlog_to_hipdnn_severity(msg.level);

        _callback_fn(severity, _udata, formatted_str.c_str());
    }

    void flush_() override {}

private:
    hipdnnCallback_t _callback_fn;
    void* _udata;
    const char* _source;
};

using callback_sink_mt = Callback_sink<std::mutex>;
using callback_sink_st = Callback_sink<spdlog::details::null_mutex>;

template <typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<spdlog::logger> create_callback_logger_mt(
    const std::string& logger_name,
    hipdnnCallback_t callback,
    void* user_data,
    const std::string& source)
{
    auto sink = std::make_shared<callback_sink_mt>(callback, user_data, source);
    return Factory::template create<spdlog::logger>(logger_name, std::move(sink));
}

// template <typename Factory = spdlog::synchronous_factory>
// inline std::shared_ptr<spdlog::logger> create_callback_logger_st(
//     const std::string& logger_name,
//     hipdnnCallback_t callback,
//     void* user_data,
//     const std::string& source)
// {
//     auto sink = std::make_shared<callback_sink_st>(callback, user_data, source);
//     return Factory::template create<spdlog::logger>(logger_name, std::move(sink));
// }

} // namespace hipdnn::logging


