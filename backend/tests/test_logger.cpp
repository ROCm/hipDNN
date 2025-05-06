// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <string>

#include <hipdnn_sdk/logging/logger.hpp>

using namespace hipdnn::logging;

class Logger_test : public ::testing::Test
{
public:
    std::string logging_area;
    std::string log_file;

    void SetUp() override
    {

        logging_area = "test_logger";
        log_file = initialize_logger_with_output_file(logging_area, "");
    }

    void TearDown() override
    {
        // when we make a file logger, it gets registered with spdlog. Need to remove it
        // so that we dont leave state for the next test.
        spdlog::drop_all();

        // always reinit a logger to std out.
        hipdnn::logging::initialize_logger_to_std_out("backend_tests");
        hipdnn::logging::set_log_level("info");

        //open the log file, if it exists, remove it
        std::ifstream log_file_stream(log_file);
        if(log_file_stream.is_open())
        {
            log_file_stream.close();
            std::remove(log_file.c_str());
        }
    }

    static std::string get_log_content(const std::string& log_file)
    {
        std::string log_content;

        std::ifstream log_file_stream(log_file);
        if(log_file_stream.is_open())
        {
            log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                               std::istreambuf_iterator<char>());
        }
        log_file_stream.close();

        return log_content;
    }

    static void wait_until_string_prints(const std::string& log_file,
                                         const std::string& string_to_wait_for,
                                         int retries)
    {
        spdlog::default_logger()->flush();
        int count = 0;
        while(count < retries)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::ifstream log_file_stream(log_file);
            if(log_file_stream.is_open())
            {
                std::string log_content((std::istreambuf_iterator<char>(log_file_stream)),
                                        std::istreambuf_iterator<char>());
                if(log_content.find(string_to_wait_for) != std::string::npos)
                {
                    break;
                }
            }
            count++;
        }
    }

    static std::string wait_until_log_content_printed(const std::string& log_file,
                                                      const std::string& expected_content)
    {
        EXPECT_FALSE(log_file.empty());
        wait_until_string_prints(log_file, expected_content, 20);
        return get_log_content(log_file);
    }

    static void verify_log_content(const std::string& log_file, const std::string& expected_content)
    {
        std::string log_content = wait_until_log_content_printed(log_file, expected_content);
        EXPECT_NE(log_content.find(expected_content), std::string::npos);
    }

    static void verify_log_content_not_found(const std::string& log_file,
                                             const std::string& expected_content)
    {
        std::string log_content = wait_until_log_content_printed(log_file, expected_content);
        EXPECT_EQ(log_content.find(expected_content), std::string::npos);
    }
};

TEST_F(Logger_test, InitializingLoggerWithFileDefaultsLoggingToOff)
{
    HIPDNN_LOG_INFO("Hi there");
    HIPDNN_LOG_WARN("Hi there");
    HIPDNN_LOG_ERROR("Hi there");

    verify_log_content_not_found(log_file, "Hi there");
}

TEST_F(Logger_test, LoggingWithLogLevelInfoPrintsAll)
{
    set_log_level("info");

    HIPDNN_LOG_INFO("logging info");
    HIPDNN_LOG_WARN("logging warning");
    HIPDNN_LOG_ERROR("logging error");

    wait_until_string_prints(log_file, "logging info", 20);
    wait_until_string_prints(log_file, "logging warning", 20);
    wait_until_string_prints(log_file, "logging error", 20);
    auto log_content = get_log_content(log_file);

    EXPECT_NE(log_content.find("logging info"), std::string::npos);
    EXPECT_NE(log_content.find("logging warning"), std::string::npos);
    EXPECT_NE(log_content.find("logging error"), std::string::npos);
}

TEST_F(Logger_test, LoggingWithLogLevelWarnPrintsWarnAndError)
{
    set_log_level("warn");

    HIPDNN_LOG_INFO("logging info");
    HIPDNN_LOG_WARN("logging warning");
    HIPDNN_LOG_ERROR("logging error");

    wait_until_string_prints(log_file, "logging warning", 20);
    wait_until_string_prints(log_file, "logging error", 20);
    auto log_content = get_log_content(log_file);

    EXPECT_EQ(log_content.find("logging info"), std::string::npos);
    EXPECT_NE(log_content.find("logging warning"), std::string::npos);
    EXPECT_NE(log_content.find("logging error"), std::string::npos);
}

TEST_F(Logger_test, LoggingWithLogLevelErrorPrintsErrorOnly)
{
    set_log_level("error");

    HIPDNN_LOG_INFO("logging info");
    HIPDNN_LOG_WARN("logging warning");
    HIPDNN_LOG_ERROR("logging error");

    wait_until_string_prints(log_file, "logging error", 20);
    auto log_content = get_log_content(log_file);

    EXPECT_EQ(log_content.find("logging info"), std::string::npos);
    EXPECT_EQ(log_content.find("logging warning"), std::string::npos);
    EXPECT_NE(log_content.find("logging error"), std::string::npos);
}

TEST_F(Logger_test, VerifyLogPattern)
{
    set_log_level("info");

    HIPDNN_LOG_INFO("logging stuff to see pattern");

    wait_until_string_prints(log_file, "logging stuff to see pattern", 20);
    auto log_content = get_log_content(log_file);
    log_content.erase(log_content.find_last_not_of(" \n\r\t") + 1); // trim off any newlines

    // Verify the pattern: [logging_area] [timestamp] [thread ID] [log level] message
    std::regex log_pattern(
        R"(\[test_logger\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] logging stuff to see pattern)");
    ASSERT_TRUE(std::regex_match(log_content, log_pattern));
}

class Environment_logger_test : public Logger_test
{
public:
    void SetUp() override
    {
        logging_area = "Env_logger_test";

        testing::internal::CaptureStdout();
        //each test has to setup the logger after env vars set.
    }

    void TearDown() override
    {
        //remove the env vars so that they dont affect other tests
        unsetenv("HIPDNN_LOG_DIR");
        unsetenv("HIPDNN_LOG_LEVEL");

        //call the base class TearDown to remove the log file
        Logger_test::TearDown();
    }
};

TEST_F(Environment_logger_test, WillLogNothingIfNoEnvVarSet)
{
    log_file = initialize_logging_based_on_environment_variables(logging_area);

    HIPDNN_LOG_ERROR("logging anything");

    std::string captured_stdout = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(log_file.empty());
    EXPECT_EQ(captured_stdout.find("logging anything"), std::string::npos);
}

TEST_F(Environment_logger_test, WillLogToStdOutIfNoDirectorySet)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    log_file = initialize_logging_based_on_environment_variables(logging_area);

    HIPDNN_LOG_INFO("logging anything");

    std::string captured_stdout = testing::internal::GetCapturedStdout();
    EXPECT_TRUE(log_file.empty());
    EXPECT_NE(captured_stdout.find("logging anything"), std::string::npos);
}

TEST_F(Environment_logger_test, WillLogNothingIfDirectorySetButNotLogLevel)
{
    std::array<char, 4096> cwd;
    getcwd(cwd.data(), cwd.size());
    setenv("HIPDNN_LOG_DIR", cwd.data(), 1);
    log_file = initialize_logging_based_on_environment_variables(logging_area);

    HIPDNN_LOG_ERROR("logging anything");

    verify_log_content_not_found(log_file, "logging anything");
    std::string captured_stdout = testing::internal::GetCapturedStdout();
    EXPECT_EQ(captured_stdout.find("logging anything"), std::string::npos);
}

TEST_F(Environment_logger_test, WillLogIfDirectorySetAndLogLevelSet)
{
    std::array<char, 4096> cwd;
    getcwd(cwd.data(), cwd.size());
    setenv("HIPDNN_LOG_DIR", cwd.data(), 1);
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    log_file = initialize_logging_based_on_environment_variables(logging_area);

    HIPDNN_LOG_INFO("logging anything");

    verify_log_content(log_file, "logging anything");
    std::string captured_stdout = testing::internal::GetCapturedStdout();
    EXPECT_EQ(captured_stdout.find("logging anything"), std::string::npos);
}