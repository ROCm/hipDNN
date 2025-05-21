// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>

#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>

#ifndef COMPONENT_NAME
#define COMPONENT_NAME "backend_tests"
#endif

class Callback_logger_test : public ::testing::Test
{
protected:
    std::string _log_file_path;
    const std::string _test_logger_name = COMPONENT_NAME;

    void SetUp() override
    {
        _log_file_path = hipdnn::logging::generate_log_file_name();
        _log_file_path = (std::filesystem::current_path() / _log_file_path).string();

        setenv("HIPDNN_LOG_FILE", _log_file_path.c_str(), 1);
        setenv("HIPDNN_LOG_LEVEL", "trace", 1);

        hipdnn::logging::initialize_callback_logging(_test_logger_name, hipdnnLoggingCallback_ext);

        auto test_logger = spdlog::get(_test_logger_name);
        ASSERT_NE(test_logger, nullptr);
        test_logger->set_level(spdlog::level::trace);
        HIPDNN_LOG_INFO("");

        auto file_writer_logger = spdlog::get("hipdnn");
        if(file_writer_logger)
        {
            file_writer_logger->flush();
        }
    }

    void TearDown() override
    {
        auto test_logger = spdlog::get(_test_logger_name);
        if(test_logger)
        {
            test_logger->flush();
        }

        auto callback_receiver_log = spdlog::get("hipdnn");
        if(callback_receiver_log)
        {
            callback_receiver_log->flush();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        if(test_logger)
        {
            spdlog::drop(_test_logger_name);
        }

        unsetenv("HIPDNN_LOG_FILE");
        unsetenv("HIPDNN_LOG_LEVEL");

        std::ifstream log_file_stream(_log_file_path);
        if(log_file_stream.is_open())
        {
            log_file_stream.close();
            if(std::remove(_log_file_path.c_str()) != 0 && HasFailure())
            {
            }
            else if(!HasFailure())
            {
            }
        }
    }

    std::string get_log_content() const
    {
        auto component_logger = spdlog::get(_test_logger_name);
        if(component_logger)
        {
            component_logger->flush();
        }

        auto file_writer_logger = spdlog::get("hipdnn");
        if(file_writer_logger)
        {
            file_writer_logger->flush();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        std::string log_content;
        std::ifstream log_file_stream(_log_file_path);
        if(log_file_stream.is_open())
        {
            log_file_stream.seekg(0, std::ios::beg);
            log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                               std::istreambuf_iterator<char>());
            log_file_stream.close();
        }
        else
        {
            ADD_FAILURE() << "Failed to open log file for reading: " << _log_file_path;
        }

        return log_content;
    }

    void verify_log_contains(const std::string& expected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << "Expected to find: \"" << expected_content << "\" in log file " << _log_file_path
            << "\nActual log content:\n"
            << log_content;
    }

    void verify_log_not_contains(const std::string& unexpected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << "Expected NOT to find: \"" << unexpected_content << "\" in log file "
            << _log_file_path << "\nActual log content:\n"
            << log_content;
    }

    void verify_log_pattern(const std::string& message_text, const std::string& level) const
    {
        std::string log_content = get_log_content();

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
    auto test_logger = spdlog::get(_test_logger_name);
    ASSERT_NE(test_logger, nullptr);
    test_logger->set_level(spdlog::level::err);

    auto file_writer_logger = spdlog::get("hipdnn");
    if(file_writer_logger)
    {
        file_writer_logger->set_level(spdlog::level::err);
    }

    HIPDNN_LOG_INFO("This info should not appear");
    HIPDNN_LOG_WARN("This warning should not appear");
    HIPDNN_LOG_ERROR("This error should appear");

    verify_log_not_contains("This info should not appear");
    verify_log_not_contains("This warning should not appear");
    verify_log_contains("This error should appear");
    verify_log_pattern("This error should appear", "error");

    test_logger->set_level(spdlog::level::trace);
    if(file_writer_logger)
    {
        file_writer_logger->set_level(spdlog::level::trace);
    }
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

    ASSERT_FALSE(matched_line.empty())
        << "Test message not found in log: " << test_message << "\nLog content:\n"
        << log_content;

    std::regex pattern_regex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[)"
        + _test_logger_name + R"(\] )" + test_message);

    EXPECT_TRUE(std::regex_match(matched_line, pattern_regex))
        << "Log line doesn't match expected pattern from generate_pattern_string.\n"
        << "Expected pattern: [timestamp] [tid thread_id] [level] [" << _test_logger_name
        << "] message\n"
        << "Actual line: " << matched_line;
}