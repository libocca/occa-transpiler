
#include "load_test_suites.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include "common/data_directory.h"
#include "oklt/core/config.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace oklt::tests {

std::vector<std::string> loadTestsSuite() {
  fs::path suitePath = DataRootHolder::instance().suitePath / "suite.json";

  if (!fs::exists(suitePath)) {
    // Test suites map file was not found
    return {};
  }
  json suite;
  try {
    std::ifstream suiteFile(suitePath);
    suite = json::parse(suiteFile);
  } catch (const std::exception& ex) {
    return {};
  }

  std::vector<std::string> resultCases;
  for (const auto& jsonPath : suite) {
    resultCases.push_back(
      (DataRootHolder::instance().suitePath / jsonPath.get<std::string>()).string());
  }
  return resultCases;
}
}  // namespace oklt::tests
