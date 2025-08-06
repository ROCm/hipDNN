// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "callback_types.h"
#include "formatting.hpp"
#include <functional>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/synchronous_factory.h>
#include <spdlog/sinks/base_sink.h>
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
        return HIPDNN_SEV_WARN;
    case spdlog::level::off:
        return HIPDNN_SEV_OFF;
    default:
        return HIPDNN_SEV_INFO;
    }
}

// This warning is for cross-compiler compatibility, so it's safe to ignore since we only use Clang.
// Moreover, MSVC is likely the incompatible compiler here.
// NOLINTBEGIN(portability-template-virtual-member-function)
template <typename Mutex>
class Callback_sink final : public spdlog::sinks::base_sink<Mutex>
{
public:
    explicit Callback_sink(hipdnnCallback_t callback)
        : _callback_fn{callback}
    {
    }
    ~Callback_sink() override = default;

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        if(_callback_fn == nullptr)
        {
            return;
        }

        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        std::string formatted_str(formatted.data(), formatted.size());

        while(!formatted_str.empty()
              && (formatted_str.back() == '\n' || formatted_str.back() == '\r'))
        {
            formatted_str.pop_back();
        }

        hipdnnSeverity_t severity = spdlog_to_hipdnn_severity(msg.level);

        _callback_fn(severity, formatted_str.c_str());
    }

    void flush_() override {}

private:
    hipdnnCallback_t _callback_fn;
};
// NOLINTEND(portability-template-virtual-member-function)

using callback_sink_mt = Callback_sink<std::mutex>;

inline std::shared_ptr<spdlog::logger> create_async_callback_logger_mt(hipdnnCallback_t callback,
                                                                       const std::string& source)
{
    auto sink = std::make_shared<hipdnn::logging::callback_sink_mt>(callback);
    auto logger = std::make_shared<spdlog::async_logger>(
        source, sink, spdlog::thread_pool(), spdlog::async_overflow_policy::block);
    logger->set_pattern(generate_pattern_string(source));
    logger->flush_on(spdlog::level::info);
    return logger;
}

template <typename Factory = spdlog::synchronous_factory>
inline std::shared_ptr<spdlog::logger> create_callback_logger_mt(hipdnnCallback_t callback,
                                                                 const std::string& source)
{
    auto logger = Factory::template create<hipdnn::logging::callback_sink_mt>(source, callback);
    logger->set_pattern(generate_pattern_string(source));
    return logger;
}

} // namespace hipdnn::logging
