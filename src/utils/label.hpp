#pragma once

#include <string>
#include <vector>


class Label : public std::vector<std::string>
{
public:
    Label() = default;
    explicit Label(const std::string& filepath);

    void Load(const std::string& filepath);
    std::string operator[] (size_type idx) const;
};
