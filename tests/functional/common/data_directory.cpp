#include "data_directory.h"
#include <string_view>
#include <vector>

namespace oklt::tests {

DataRootHolder & DataRootHolder::instance() {
    static DataRootHolder holder;
    return holder;
}
}
