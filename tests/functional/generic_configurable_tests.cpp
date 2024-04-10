#include "common/data_directory.h"
#include "common/load_test_suites.h"

#include <oklt/core/error.h>
#include <oklt/core/target_backends.h>
#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>
#include <oklt/util/format.h>
#include <oklt/util/string_utils.h>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

#include <fstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace oklt::tests;

enum struct Action { NORMALIZER, TRANSPILER, NORMALIZE_AND_TRANSPILE };
enum struct Compare { SOURCE, METADATA, ERROR_MESSAGE };

tl::expected<Action, std::string> buildActionFrom(const std::string& v) {
    static const std::map<std::string, Action> actions = {
        {"normalizer", Action::NORMALIZER},
        {"transpiler", Action::TRANSPILER},
        {"normalize_and_transpile", Action::NORMALIZE_AND_TRANSPILE},
    };
    auto it = actions.find(oklt::util::toLower(v));
    if (it != actions.cend()) {
        return it->second;
    }
    return tl::unexpected<std::string>("Unknown action is used");
}

tl::expected<Compare, std::string> buildCompareFrom(const std::string& v) {
    static const std::map<std::string, Compare> compares = {
        {"source", Compare::SOURCE},
        {"metadata", Compare::METADATA},
        {"error_message", Compare::ERROR_MESSAGE},
    };
    auto it = compares.find(oklt::util::toLower(v));
    if (it != compares.cend()) {
        return it->second;
    }
    return tl::unexpected<std::string>("Unknown compare is used");
}

struct NormalizeActionConfig {
    std::filesystem::path source;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(NormalizeActionConfig, source)
    oklt::UserInput build(const fs::path& dataDir) const;
};

oklt::UserInput NormalizeActionConfig::build(const fs::path& dataDir) const {
    auto sourceFullPath = dataDir / source;
    std::ifstream sourceFile{sourceFullPath};
    std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};
    return oklt::UserInput{.backend = oklt::TargetBackend::CUDA,
                           .sourceCode = std::move(sourceCode)};
}

struct TranspileActionConfig {
    std::string backend;
    std::filesystem::path source;
    std::vector<std::filesystem::path> mutable includes;
    std::vector<std::string> mutable defs;
    std::filesystem::path launcher;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(TranspileActionConfig, backend, source, includes, defs, launcher)
    oklt::UserInput build(const fs::path& dataDir) const;
};

oklt::UserInput TranspileActionConfig::build(const fs::path& dataDir) const {
    auto expectedBackend = oklt::backendFromString(backend);
    if (!expectedBackend) {
        throw std::logic_error(expectedBackend.error());
    }
    auto sourceFullPath = dataDir / source;
    std::ifstream sourceFile{sourceFullPath};
    std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};

    return oklt::UserInput{.backend = expectedBackend.value(),
                           .astProcType = oklt::AstProcessorType::OKL_WITH_SEMA,
                           .sourceCode = std::move(sourceCode),
                           .sourcePath = std::move(sourceFullPath),
                           .inlcudeDirectories = includes,
                           .defines = defs};
}

namespace {
void compareError(const std::string& sourceFilePath,
                  const oklt::UserResult& res,
                  const std::string& refErrorMessage) {
    // TODO: currently compare only first error to the whole file. There can be
    // multiple errors
    EXPECT_EQ(res.error().empty(), refErrorMessage.empty())
        << "No error when expected, or vice versa";
    auto normalizeError = res.error().front();
    auto errorMessage = normalizeError.desc;

    // Since path to file in error message depends on pwd, we have to replace it with just filename
    // Error must start with sourceFilePath
    auto filename = fs::path(sourceFilePath).filename().string();
    if (errorMessage.find(sourceFilePath) == 0) {
        errorMessage.replace(0, sourceFilePath.size(), filename);
    }

    if (errorMessage != refErrorMessage) {
        std::cout << "hello\n";
    }

    EXPECT_EQ(errorMessage, refErrorMessage)
        << "Error message is different for file: " << sourceFilePath;
}
}  // namespace

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

        Compare cmp = Compare::SOURCE;
        it = testCase.find("compare");
        if (it != testCase.end()) {
            auto expectedCmp = buildCompareFrom(it->get<std::string>());
            if (!expectedCmp) {
                EXPECT_TRUE(false) << "Wrong compare type" << std::endl;
            }
            cmp = expectedCmp.value();
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
                auto input = conf.build(dataDir);
                auto normalizeResult = oklt::normalize(input);
                if (!normalizeResult && cmp != Compare::ERROR_MESSAGE) {
                    EXPECT_TRUE(false) << "File: " << conf.source << std::endl
                                       << "Normalize error occur" << std::endl;
                }

                std::ifstream referenceFile(referencePath);
                std::string reference{std::istreambuf_iterator<char>(referenceFile), {}};
                switch (cmp) {
                    case Compare::SOURCE: {
                        std::string formatedReference = oklt::format(reference);
                        std::string normalizedSource =
                            oklt::format(normalizeResult.value().normalized.sourceCode);
                        EXPECT_EQ(formatedReference, normalizedSource)
                            << fmt::format("Failed compare {} with generated normalized source",
                                           referencePath.string());
                        break;
                    }
                    case Compare::METADATA: {
                        EXPECT_EQ(reference, normalizeResult.value().normalized.metadataJson)
                            << fmt::format(
                                   "Failed compare {} with generated normalization metadataJson",
                                   referencePath.string());
                        break;
                    }
                    case Compare::ERROR_MESSAGE: {
                        compareError(input.sourcePath, normalizeResult, reference);
                        break;
                    }
                }

            } break;
            case Action::TRANSPILER: {
                auto actionConfig = testCase.find("action_config");
                if (actionConfig == testCase.cend()) {
                    GTEST_SKIP_("Can't get action_config field");
                    continue;
                }
                auto conf = actionConfig->get<TranspileActionConfig>();
                auto input = conf.build(dataDir);
                auto transpileResult = oklt::transpile(input);

                if (!transpileResult && cmp != Compare::ERROR_MESSAGE) {
                    std::string error;
                    for (const auto& e : transpileResult.error()) {
                        error += e.desc + "\n";
                    }
                    EXPECT_TRUE(false) << "File: " << conf.source << std::endl
                                       << "Transpile error:" << error << std::endl;
                }

                std::ifstream referenceFile(referencePath);
                std::string reference{std::istreambuf_iterator<char>(referenceFile), {}};
                switch (cmp) {
                    case Compare::SOURCE: {
                        std::string formatedReference = oklt::format(reference);
                        std::string transpiledSource =
                            oklt::format(transpileResult.value().kernel.sourceCode);
                        EXPECT_EQ(formatedReference, transpiledSource) << fmt::format(
                            "Failed compare {} with generated normalization metadataJson",
                            referencePath.string());

                        break;
                    }
                    case Compare::METADATA: {
                        EXPECT_EQ(reference, transpileResult.value().kernel.metadataJson)
                            << fmt::format("Failed compare {} with generated kernel metadataJson",
                                           referencePath.string());

                        break;
                    }
                    case Compare::ERROR_MESSAGE: {
                        compareError(input.sourcePath, transpileResult, reference);
                        break;
                    }
                }

            } break;
            case Action::NORMALIZE_AND_TRANSPILE: {
                auto actionConfig = testCase.find("action_config");
                if (actionConfig == testCase.cend()) {
                    GTEST_SKIP_("Can't get action_config field");
                    continue;
                }
                auto conf = actionConfig->get<TranspileActionConfig>();
                auto input = conf.build(dataDir);
                auto transpileResult = oklt::normalizeAndTranspile(input);

                if (!transpileResult && cmp != Compare::ERROR_MESSAGE) {
                    std::string error;
                    for (const auto& e : transpileResult.error()) {
                        error += e.desc + "\n";
                    }
                    EXPECT_TRUE(false) << "File: " << conf.source << std::endl
                                       << "Normalize & Transpile error:" << error << std::endl;
                }

                std::ifstream referenceFile(referencePath);
                std::string reference{std::istreambuf_iterator<char>(referenceFile), {}};
                switch (cmp) {
                    case Compare::SOURCE: {
                        std::string formatedReference = oklt::format(reference);
                        std::string transpiledSource =
                            oklt::format(transpileResult.value().kernel.sourceCode);
                        EXPECT_EQ(formatedReference, transpiledSource) << fmt::format(
                            "Failed compare {} with generated normalized and transpiled",
                            referencePath.string());
                        break;
                    }
                    case Compare::METADATA: {
                        EXPECT_EQ(reference, transpileResult.value().kernel.metadataJson)
                            << fmt::format("Failed compare {} with generated kernel metadataJson",
                                           referencePath.string());

                        break;
                    }
                    case Compare::ERROR_MESSAGE: {
                        compareError(input.sourcePath, transpileResult, reference);
                        break;
                    }
                }
                break;
            }
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
