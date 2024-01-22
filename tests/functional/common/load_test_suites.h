#include <string>
#include <vector>
#include <list>
#include <filesystem>

#include "oklt/core/config.h"

namespace oklt
{
namespace tests
{

struct GenericSuite {
    TRANSPILER_TYPE backendType;
    std::list<std::filesystem::path> testCaseFiles;
};

std::vector<GenericSuite> loadTestsSuite();

                                          
}  // namespace tests
}  // namespace oklt
 
