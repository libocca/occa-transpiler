#include "common/data_directory.h"
#include "common/load_test_suites.h"
#include "oklt/pipeline/stages/transpiler/transpiler.h"
#include "oklt/pipeline/stages/normalizer/normalizer.h"
#include "oklt/pipeline/normalize_and_transpile.h"
#include "oklt/core/config.h"
#include "oklt/util/string_utils.h"
#include "oklt/core/utils/format.h"
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace oklt::tests;


enum struct Action {
  NORMALIZER,
  TRANSPILER,
  NORMALIZE_AND_TRANSPILE
};

tl::expected<Action, std::string> buildActionFrom(const std::string &v) {
  static const std::map<std::string, Action> actions = {
    {"normalizer", Action::NORMALIZER},
    {"transpilier", Action::TRANSPILER},
    {"transpile_and_normalize", Action::NORMALIZE_AND_TRANSPILE},
  };
  auto it = actions.find(oklt::util::toLower(v));
  if(it != actions.cend()) {
    return it->second;
  }
  return tl::unexpected<std::string>("Unknown action is used");
}

struct NormalizeActionConfig {
  std::filesystem::path source;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(NormalizeActionConfig,
                                 source)
};

struct TranspileActionConfig {
  std::string backend;
  std::filesystem::path source;
  std::list<std::filesystem::path> includes;
  std::list<std::string> defs;
  std::filesystem::path launcher;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TranspileActionConfig,
                                 backend,
                                 source,
                                 includes,
                                 defs,
                                 launcher)
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
        auto it = testCase.find("action");
        if(it == testCase.cend()) {
          GTEST_SKIP_("Can't get action field");
          continue;
        }
        auto expectedAction = buildActionFrom(it->get<std::string>());
        if(!expectedAction) {
          EXPECT_TRUE(false) << "Wrong action type" << std::endl;
        }
        auto referencePath = testCase["reference"].get<std::filesystem::path>();
        referencePath = dataDir / referencePath;

        switch(expectedAction.value()) {
          case Action::NORMALIZER: {
            auto actionConfig = testCase.find("action_config");
            if(actionConfig == testCase.cend()) {
              GTEST_SKIP_("Can't get action_config field");
              continue;
            }
            auto conf = actionConfig->get<NormalizeActionConfig>();

            auto sourceFullPath = dataDir / conf.source;
            std::ifstream sourceFile {sourceFullPath};
            std::string sourceCode {std::istreambuf_iterator<char>(sourceFile), {}};
            //INFO: does not metter here
            oklt::TranspilerSession session {oklt::TRANSPILER_TYPE::CUDA};
            auto normalizeResult = oklt::normalize(oklt::NormalizerInput {sourceCode}, session);
            if(!normalizeResult) {
              EXPECT_TRUE(false) << "Normalize error occur" << std::endl;
            }

            std::ifstream ifs(referencePath);
            std::string referenceSource {std::istreambuf_iterator<char>(sourceFile), {}};
            std::string formatedReference = oklt::format(referenceSource);
            std::string normalizedSource = oklt::format(normalizeResult.value().cppSource);
            EXPECT_EQ(formatedReference, normalizedSource);
          }break;
          case Action::TRANSPILER: {
            auto actionConfig = testCase.find("action_config");
            if(actionConfig == testCase.cend()) {
              GTEST_SKIP_("Can't get action_config field");
              continue;
            }
            auto conf = actionConfig->get<TranspileActionConfig>();

            auto expectedBackend = oklt::backendFromString(conf.backend);
            if(!expectedBackend) {
              EXPECT_TRUE(false) << expectedBackend.error() << std::endl;
              continue;
            }

            auto sourceFullPath = dataDir / conf.source;
            std::ifstream sourceFile {sourceFullPath};
            std::string sourceCode {std::istreambuf_iterator<char>(sourceFile), {}};
            auto transpileResult = oklt::transpile(oklt::TranspilerInput{
              .sourceCode = sourceCode,
              .sourcePath =sourceFullPath,
              .inlcudeDirectories = conf.includes,
              .defines = conf.defs,
              .targetBackend = expectedBackend.value()
            });

            if(!transpileResult) {
              std::string error;
              for(const auto &e: transpileResult.error()) {
                error += e.desription + "\n";
              }
              EXPECT_TRUE(false) << "Transpile error:" << error << std::endl;
            }

            std::ifstream ifs(referencePath);
            std::string referenceSource {std::istreambuf_iterator<char>(sourceFile), {}};
            std::string formatedReference = oklt::format(referenceSource);
            std::string transpiledSource = oklt::format(transpileResult.value().kernel.outCode);
            EXPECT_EQ(formatedReference, transpiledSource);
          }break;
          case Action::NORMALIZE_AND_TRANSPILE:
          {

          }break;
        }
    }
}


INSTANTIATE_TEST_SUITE_P(
    GenericSuiteTests,
    GenericTest,
    testing::ValuesIn( loadTestsSuite() ));
