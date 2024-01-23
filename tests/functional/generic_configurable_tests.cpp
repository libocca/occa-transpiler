#include "common/data_directory.h"
#include "common/load_test_suites.h"
#include "oklt/core/transpile.h"
#include "oklt/core/utils/format.h"
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace oklt::tests;


struct TestCase {
  std::filesystem::path config;
  std::filesystem::path source;
  std::filesystem::path reference;
  std::filesystem::path launcher;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TestCase,
                                 config,
                                 source,
                                 reference,
                                 launcher)
  TestCase & rootDataDir(const std::filesystem::path &root) {
    config = root / config;
    source = root / source;
    reference = root / reference;
    launcher = root / launcher;
    return *this;
  }
};


class GenericTest : public testing::TestWithParam<fs::path>
{};

TEST_P( GenericTest, OCCATests )
{
    auto dataDir = DataRootHolder::instance().dataRoot;
    auto suitPath = GetParam();
    std::ifstream testSuitFile(suitPath);
    json tests = json::parse( testSuitFile );

    for(const auto &testCase: tests) {
        auto test = testCase.get<TestCase>();
        test = test.rootDataDir(dataDir);

        std::ifstream referenceFile(test.reference);
        std::ifstream configFile(tests);
        std::string config {std::istreambuf_iterator<char>(configFile), {}};

        //TODO: change to Pipeline builder
        auto input = oklt::make_transpile_input(test.source, config);
        if(!input) {
          EXPECT_TRUE(false) << "Can't build transpile input for : " << test.config << std::endl;
        }
        auto result = oklt::transpile(input.value());
        if(!result) {
         std::string multiErrorValue;
         for(const auto &error: result.error()) {
           multiErrorValue += error.desription + "\n";
         }
         EXPECT_TRUE(false) << multiErrorValue;
        }
        auto transpiledResult = result.value();
        std::string referenceSource {std::istream_iterator<char>(referenceFile), {}};
        std::string formatedReference = oklt::format(referenceSource);
        std::string transpiledSource = oklt::format(transpiledResult.kernel.outCode);
        EXPECT_EQ(formatedReference, transpiledSource);
    }
}


INSTANTIATE_TEST_SUITE_P(
    GenericSuiteTests,
    GenericTest,
    testing::ValuesIn( loadTestsSuite() ));
