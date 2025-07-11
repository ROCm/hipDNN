// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <array>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <unistd.h>

#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>

#ifndef COMPONENT_NAME
#define COMPONENT_NAME "backend_tests"
#endif

class Callback_logger_test : public ::testing::Test
{
protected:
    const std::string _test_logger_name = COMPONENT_NAME;
    std::string _log_file_path;
    std::array<int, 2> _stderr_pipe;
    int _old_stderr;

    void SetUp() override
    {
        // pipe stderr to capture log output
        _old_stderr = dup(STDERR_FILENO);
        ASSERT_NE(_old_stderr, -1);
        ASSERT_EQ(pipe(_stderr_pipe.data()), 0);
        ASSERT_NE(dup2(_stderr_pipe[1], STDERR_FILENO), -1);
        ASSERT_EQ(close(_stderr_pipe[1]), 0);

        unsetenv("HIPDNN_LOG_FILE");
        setenv("HIPDNN_LOG_LEVEL", "info", 1);

        spdlog::drop_all();

        hipdnn::logging::initialize_callback_logging(_test_logger_name, hipdnnLoggingCallback_ext);

        auto test_logger = spdlog::get(_test_logger_name);
        ASSERT_NE(test_logger, nullptr);
        test_logger->set_level(spdlog::level::info);
    }

    void TearDown() override
    {
        spdlog::apply_all([&](const std::shared_ptr<spdlog::logger>& l) { l->flush(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        fflush(stderr);
        dup2(_old_stderr, STDERR_FILENO);
        close(_old_stderr);
        close(_stderr_pipe[0]);

        spdlog::drop_all();
        unsetenv("HIPDNN_LOG_FILE");
        unsetenv("HIPDNN_LOG_LEVEL");

        if(!_log_file_path.empty())
        {
            std::remove(_log_file_path.c_str());
        }
    }

    std::string get_stderr_content()
    {
        spdlog::apply_all([&](const std::shared_ptr<spdlog::logger>& l) { l->flush(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content;
        std::array<char, 4096> buffer;
        ssize_t bytes_read;

        // make read non-blocking
        fcntl(_stderr_pipe[0], F_SETFL, O_NONBLOCK);

        while((bytes_read = read(_stderr_pipe[0], buffer.data(), buffer.size() - 1)) > 0)
        {
            buffer[static_cast<size_t>(bytes_read)] = '\0';
            content += buffer.data();
        }
        return content;
    }

    void verify_stderr_contains(const std::string& expected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << "Expected to find: \"" << expected_content << "\" in stderr."
            << "\nActual stderr content:\n"
            << log_content;
    }

    void verify_stderr_not_contains(const std::string& unexpected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << "Expected NOT to find: \"" << unexpected_content << "\" in stderr."
            << "\nActual stderr content:\n"
            << log_content;
    }

    void verify_stderr_pattern(const std::string& log_content,
                               const std::string& message_text,
                               const std::string& level)
    {
        std::regex pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[)"
                           + level + R"(\] \[)" + _test_logger_name + R"(\] )" + message_text);

        EXPECT_TRUE(std::regex_search(log_content, pattern))
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

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find(test_message), std::string::npos);
    verify_stderr_pattern(log_content, test_message, "info");
}

TEST_F(Callback_logger_test, WarnMessageIsCorrectlyLogged)
{
    std::string test_message = "Test warning message";
    HIPDNN_LOG_WARN(test_message);

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find(test_message), std::string::npos);
    verify_stderr_pattern(log_content, test_message, "warning");
}

TEST_F(Callback_logger_test, ErrorMessageIsCorrectlyLogged)
{
    std::string test_message = "Test error message";
    HIPDNN_LOG_ERROR(test_message);

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find(test_message), std::string::npos);
    verify_stderr_pattern(log_content, test_message, "error");
}

TEST_F(Callback_logger_test, FormattedMessagesAreCorrectlyLogged)
{
    int value = 42;
    std::string text = "formatted";

    HIPDNN_LOG_INFO("Test {} message with value {}", text, value);

    std::string expected_content = "Test formatted message with value 42";
    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find(expected_content), std::string::npos);
    verify_stderr_pattern(log_content, expected_content, "info");
}

TEST_F(Callback_logger_test, LogLevelsAreRespected)
{
    auto test_logger = spdlog::get(_test_logger_name);
    ASSERT_NE(test_logger, nullptr);
    test_logger->set_level(spdlog::level::err);

    HIPDNN_LOG_INFO("This info should not appear");
    HIPDNN_LOG_WARN("This warning should not appear");
    HIPDNN_LOG_ERROR("This error should appear");

    std::string log_content = get_stderr_content();
    EXPECT_EQ(log_content.find("This info should not appear"), std::string::npos);
    EXPECT_EQ(log_content.find("This warning should not appear"), std::string::npos);
    EXPECT_NE(log_content.find("This error should appear"), std::string::npos);

    test_logger->set_level(spdlog::level::info);
}

TEST_F(Callback_logger_test, MultipleMessagesAreLogged)
{
    HIPDNN_LOG_INFO("First message");
    HIPDNN_LOG_INFO("Second message");
    HIPDNN_LOG_INFO("Third message");

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find("First message"), std::string::npos);
    EXPECT_NE(log_content.find("Second message"), std::string::npos);
    EXPECT_NE(log_content.find("Third message"), std::string::npos);
}

TEST_F(Callback_logger_test, LoggingToFile)
{
    TearDown();
    _log_file_path = "callback_test_file.log";
    setenv("HIPDNN_LOG_FILE", _log_file_path.c_str(), 1);
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

    spdlog::drop_all();
    hipdnn::logging::initialize_callback_logging(_test_logger_name, hipdnnLoggingCallback_ext);
    auto test_logger = spdlog::get(_test_logger_name);
    ASSERT_NE(test_logger, nullptr);
    test_logger->set_level(spdlog::level::info);

    std::string test_message = "This message goes to the log file.";
    HIPDNN_LOG_INFO(test_message);

    spdlog::apply_all([&](const std::shared_ptr<spdlog::logger>& l) { l->flush(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::string log_content;
    std::ifstream log_file_stream(_log_file_path);
    ASSERT_TRUE(log_file_stream.is_open()) << "Log file was not created: " << _log_file_path;

    log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                       std::istreambuf_iterator<char>());
    log_file_stream.close();

    EXPECT_NE(log_content.find(test_message), std::string::npos)
        << "Expected to find message in log file " << _log_file_path << "\nActual log content:\n"
        << log_content;
}