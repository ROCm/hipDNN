// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>

#include <hipdnn_backend.h>
#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>

class Callback_logger_test : public ::testing::Test
{
protected:
    std::string _log_file;
    const std::string _test_logger_name = "test_callback_logger";
    std::shared_ptr<spdlog::logger> _original_default_logger;

    void SetUp() override
    {
        _original_default_logger = spdlog::default_logger();

        _log_file = hipdnn::logging::generate_log_file_name();
        setenv("HIPDNN_LOG_FILE", _log_file.c_str(), 1);
        setenv("HIPDNN_LOG_LEVEL", "info", 1);

        hipdnn::logging::initialize_callback_logging(
            _test_logger_name, hipdnnLoggingCallback_ext, nullptr);

        // Ensure logs of all levels are captured during tests
        spdlog::set_level(spdlog::level::trace);
    }

    void TearDown() override
    {
        // Restore original logger state
        if(spdlog::get(_test_logger_name))
        {
            spdlog::drop(_test_logger_name);
        }

        spdlog::set_default_logger(_original_default_logger);

        unsetenv("HIPDNN_LOG_FILE");
        unsetenv("HIPDNN_LOG_LEVEL");

        std::ifstream log_file_stream(_log_file);
        if(log_file_stream.is_open())
        {
            log_file_stream.close();
            std::remove(_log_file.c_str());
        }
    }

    std::string get_log_content() const
    {
        spdlog::default_logger_raw()->flush();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string log_content;
        std::ifstream log_file_stream(_log_file);
        if(log_file_stream.is_open())
        {
            log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                               std::istreambuf_iterator<char>());
            log_file_stream.close();
        }
        return log_content;
    }

    void verify_log_contains(const std::string& expected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << "Expected to find: \"" << expected_content << "\" in log file " << _log_file
            << "\nActual log content:\n"
            << log_content;
    }

    void verify_log_not_contains(const std::string& unexpected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << "Expected NOT to find: \"" << unexpected_content << "\" in log file " << _log_file
            << "\nActual log content:\n"
            << log_content;
    }

    void verify_log_pattern(const std::string& message_text, const std::string& level) const
    {
        std::string log_content = get_log_content();

        // [timestamp] [tid thread_id] [level] [logger_name] message
        std::regex pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[)"
                           + level + R"(\] \[)" + _test_logger_name + R"(\] )" + message_text);

        bool pattern_matched = false;
        std::istringstream log_stream(log_content);
        std::string line;

        while(std::getline(log_stream, line))
        {
            if(std::regex_match(line, pattern))
            {
                pattern_matched = true;
                break;
            }
        }

        EXPECT_TRUE(pattern_matched)
            << "Expected log pattern not found for message: " << message_text
            << "\nExpected pattern: [timestamp] [tid thread_id] [" << level << "] ["
            << _test_logger_name << "] " << message_text << "\nLog content:\n"
            << log_content;
    }
};

TEST_F(Callback_logger_test, InfoMessageIsCorrectlyLogged)
{
    std::string test_message = "Test info message";
    HIPDNN_LOG_INFO(test_message);

    verify_log_contains(test_message);
    verify_log_pattern(test_message, "info");
}

TEST_F(Callback_logger_test, WarnMessageIsCorrectlyLogged)
{
    std::string test_message = "Test warning message";
    HIPDNN_LOG_WARN(test_message);

    verify_log_contains(test_message);
    verify_log_pattern(test_message, "warning");
}

TEST_F(Callback_logger_test, ErrorMessageIsCorrectlyLogged)
{
    std::string test_message = "Test error message";
    HIPDNN_LOG_ERROR(test_message);

    verify_log_contains(test_message);
    verify_log_pattern(test_message, "error");
}

TEST_F(Callback_logger_test, FormattedMessagesAreCorrectlyLogged)
{
    int value = 42;
    std::string text = "formatted";

    HIPDNN_LOG_INFO("Test {} message with value {}", text, value);

    std::string expected_content = "Test formatted message with value 42";
    verify_log_contains(expected_content);
    verify_log_pattern(expected_content, "info");
}

TEST_F(Callback_logger_test, LogLevelsAreRespected)
{
    spdlog::set_level(spdlog::level::err);

    HIPDNN_LOG_INFO("This info should not appear");
    HIPDNN_LOG_WARN("This warning should not appear");
    HIPDNN_LOG_ERROR("This error should appear");

    verify_log_not_contains("This info should not appear");
    verify_log_not_contains("This warning should not appear");
    verify_log_contains("This error should appear");
    verify_log_pattern("This error should appear", "error");

    spdlog::set_level(spdlog::level::trace);
}

TEST_F(Callback_logger_test, MultipleMessagesAreLogged)
{
    HIPDNN_LOG_INFO("First message");
    HIPDNN_LOG_INFO("Second message");
    HIPDNN_LOG_INFO("Third message");

    verify_log_contains("First message");
    verify_log_contains("Second message");
    verify_log_contains("Third message");
    verify_log_pattern("First message", "info");
    verify_log_pattern("Second message", "info");
    verify_log_pattern("Third message", "info");
}

TEST_F(Callback_logger_test, VerifyLogPatternMatchesSpecification)
{
    std::string test_message = "Pattern verification test message";
    HIPDNN_LOG_INFO(test_message);

    std::string log_content = get_log_content();

    std::istringstream log_stream(log_content);
    std::string line;
    std::string matched_line;

    while(std::getline(log_stream, line))
    {
        if(line.find(test_message) != std::string::npos)
        {
            matched_line = line;
            break;
        }
    }

    ASSERT_FALSE(matched_line.empty()) << "Test message not found in log: " << test_message;

    // "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] [%n] %v" (formatting.hpp)
    std::regex pattern_regex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[)"
        + _test_logger_name + R"(\] )" + test_message);

    EXPECT_TRUE(std::regex_match(matched_line, pattern_regex))
        << "Log line doesn't match expected pattern from generate_pattern_string.\n"
        << "Expected pattern: [timestamp] [tid thread_id] [level] [" << _test_logger_name
        << "] message\n"
        << "Actual line: " << matched_line;
}