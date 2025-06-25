#include "label.hpp"

#include <fstream>
#include <cstring>
#include <cstdio>


Label::Label(const std::string& filepath)
{
    Load(filepath);
}

void Label::Load(const std::string& filepath)
{
    std::fstream ifs(filepath, std::ios::in);
    if (!ifs.good()) {
        std::printf("open label file failed");
        return;
    }

    if (!this->empty()) {
        this->clear();
    }

    while (!ifs.eof()) {
        char line[128] = {0};
        ifs.getline(line, 128, '\n');
        if (std::strlen(line) > 0) {
            this->emplace_back(line);
        }
    }

    ifs.close();
}

std::string Label::operator[] (size_type idx) const
{
    if (this->empty() || idx >= this->size()) {
        return std::to_string(idx);
    }
    return this->at(idx);
}
