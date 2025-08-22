#pragma once

#include <filesystem>

class Temp_dir
{
    std::filesystem::path _path;

public:
    Temp_dir(std::filesystem::path path)
    {
        if(std::filesystem::create_directory(path))
        {
            _path = std::move(path);
        }
        else
        {
            std::runtime_error("Temp_dir: Directory already exists");
        }
    }
    const std::filesystem::path& path() const
    {
        return _path;
    }

    Temp_dir(const Temp_dir&) = delete;
    Temp_dir& operator=(const Temp_dir&) = delete;
    Temp_dir(Temp_dir&&) = default;
    Temp_dir& operator=(Temp_dir&&) = default;
    ~Temp_dir()
    {
        if(!_path.empty())
        {
            std::filesystem::remove_all(_path);
        }
    }
};
