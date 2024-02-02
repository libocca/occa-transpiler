#include "common/data_directory.h"
#include "common/load_test_suites.h"

#include <oklt/core/error.h>
#include <oklt/core/target_backends.h>
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <oklt/core/utils/format.h>
#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>
#include <oklt/util/string_utils.h>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>
#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace oklt::tests;

enum struct Action { NORMALIZER, TRANSPILER, NORMALIZE_AND_TRANSPILE };

tl::expected<Action, std::string> buildActionFrom(const std::string& v) {
  static const std::map<std::string, Action> actions = {
    {"normalizer", Action::NORMALIZER},
    {"transpilier", Action::TRANSPILER},
    {"transpile_and_normalize", Action::NORMALIZE_AND_TRANSPILE},
  };
  auto it = actions.find(oklt::util::toLower(v));
  if (it != actions.cend()) {
    return it->second;
  }
  return tl::unexpected<std::string>("Unknown action is used");
}

struct NormalizeActionConfig {
  std::filesystem::path source;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(NormalizeActionConfig, source)
  std::shared_ptr<oklt::TranspilerSession> build(const fs::path& dataDir) const;
};

std::shared_ptr<oklt::TranspilerSession> NormalizeActionConfig::build(
  const fs::path& dataDir) const {
  auto sourceFullPath = dataDir / source;
  std::ifstream sourceFile{sourceFullPath};
  std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};
  return std::make_shared<oklt::TranspilerSession>(oklt::TargetBackend::CUDA, sourceCode);
}

struct TranspileActionConfig {
  std::string backend;
  std::filesystem::path source;
  std::vector<std::filesystem::path> mutable includes;
  std::vector<std::string> mutable defs;
  std::filesystem::path launcher;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TranspileActionConfig, backend, source, includes, defs, launcher)
  std::shared_ptr<oklt::TranspilerSession> build(const fs::path& dataDir) const;
};

std::shared_ptr<oklt::TranspilerSession> TranspileActionConfig::build(
  const fs::path& dataDir) const {
  auto expectedBackend = oklt::backendFromString(backend);
  if (!expectedBackend) {
    throw std::logic_error(expectedBackend.error());
  }
  auto sourceFullPath = dataDir / source;
  std::ifstream sourceFile{sourceFullPath};
  std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};

  return std::make_shared<oklt::TranspilerSession>(oklt::TranspilerSession::UserInput{
    expectedBackend.value(), std::move(sourceCode), std::move(sourceFullPath), includes, defs});
}

class GenericTest : public testing::TestWithParam<std::string> {};

TEST_P(GenericTest, OCCATests) {
  auto dataDir = DataRootHolder::instance().dataRoot;
  fs::path suitPath = GetParam();
  std::ifstream testSuitFile(suitPath);
  json tests = json::parse(testSuitFile);

  for (const auto& testCase : tests) {
    auto it = testCase.find("action");
    if (it == testCase.cend()) {
      GTEST_SKIP_("Can't get action field");
      continue;
    }
    auto expectedAction = buildActionFrom(it->get<std::string>());
    if (!expectedAction) {
      EXPECT_TRUE(false) << "Wrong action type" << std::endl;
    }
    auto referencePath = testCase["reference"].get<std::filesystem::path>();
    referencePath = dataDir / referencePath;

    switch (expectedAction.value()) {
      case Action::NORMALIZER: {
        auto actionConfig = testCase.find("action_config");
        if (actionConfig == testCase.cend()) {
          GTEST_SKIP_("Can't get action_config field");
          continue;
        }
        auto conf = actionConfig->get<NormalizeActionConfig>();
        auto normalizeResult = oklt::normalize(conf.build(dataDir));
        if (!normalizeResult) {
          EXPECT_TRUE(false) << "Normalize error occur" << std::endl;
        }

        std::ifstream referenceFile(referencePath);
        std::string referenceSource{std::istreambuf_iterator<char>(referenceFile), {}};
        std::string formatedReference = oklt::format(referenceSource);
        std::string normalizedSource =
          oklt::format(normalizeResult.value()->output.normalized.outCode);
        EXPECT_EQ(formatedReference, normalizedSource);
      } break;
      case Action::TRANSPILER: {
        auto actionConfig = testCase.find("action_config");
        if (actionConfig == testCase.cend()) {
          GTEST_SKIP_("Can't get action_config field");
          continue;
        }
        auto conf = actionConfig->get<TranspileActionConfig>();
        auto transpileResult = oklt::transpile(conf.build(dataDir));

        if (!transpileResult) {
          std::string error;
          for (const auto& e : transpileResult.error()) {
            error += e.desc + "\n";
          }
          EXPECT_TRUE(false) << "Transpile error:" << error << std::endl;
        }

        std::ifstream referenceFile(referencePath);
        std::string referenceSource{std::istreambuf_iterator<char>(referenceFile), {}};
        std::string formatedReference = oklt::format(referenceSource);
        std::string transpiledSource = oklt::format(transpileResult.value()->output.kernel.outCode);
        EXPECT_EQ(formatedReference, transpiledSource);
      } break;
      case Action::NORMALIZE_AND_TRANSPILE: {
        auto actionConfig = testCase.find("action_config");
        if (actionConfig == testCase.cend()) {
          GTEST_SKIP_("Can't get action_config field");
          continue;
        }
        auto conf = actionConfig->get<TranspileActionConfig>();
        auto transpileResult = oklt::normalizeAndTranspile(conf.build(dataDir));

        if (!transpileResult) {
          std::string error;
          for (const auto& e : transpileResult.error()) {
            error += e.desc + "\n";
          }
          EXPECT_TRUE(false) << "Normalize & Transpile error:" << error << std::endl;
        }

        std::ifstream referenceFile(referencePath);
        std::string referenceSource{std::istreambuf_iterator<char>(referenceFile), {}};
        std::string formatedReference = oklt::format(referenceSource);
        std::string transpiledSource = oklt::format(transpileResult.value()->output.kernel.outCode);
        EXPECT_EQ(formatedReference, transpiledSource);
      } break;
    }
  }
}

struct GenericConfigTestNamePrinter {
  std::string operator()(const testing::TestParamInfo<std::string>& info) const {
    std::filesystem::path fullPath(info.param);
    auto fileName = fullPath.stem();
    return oklt::util::toCamelCase(fileName.string());
  }
};

INSTANTIATE_TEST_SUITE_P(GenericSuiteTests,
                         GenericTest,
                         testing::ValuesIn(loadTestsSuite()),
                         GenericConfigTestNamePrinter());
