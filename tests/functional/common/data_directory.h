#pragma once

#include <filesystem>

namespace oklt::tests {

struct DataRootHolder {
    std::filesystem::path dataRoot;
    std::filesystem::path suitePath;
    static DataRootHolder& instance();
};
}
