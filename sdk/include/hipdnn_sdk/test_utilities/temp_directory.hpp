#pragma once

#include <filesystem>

class Temp_directory
{
    std::filesystem::path _path;

public:
    Temp_directory(std::filesystem::path path)
    {
        if(std::filesystem::create_directory(path))
        {
            _path = std::move(path);
        }
        else
        {
            throw std::runtime_error("Temp_directory: Directory already exists");
        }
    }
    const std::filesystem::path& path() const
    {
        return _path;
    }

    Temp_directory(const Temp_directory&) = delete;
    Temp_directory& operator=(const Temp_directory&) = delete;
    Temp_directory(Temp_directory&&) = default;
    Temp_directory& operator=(Temp_directory&&) = default;
    ~Temp_directory()
    {
        if(!_path.empty())
        {
            std::filesystem::remove_all(_path);
        }
    }
};
