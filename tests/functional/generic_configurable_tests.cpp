#include "common/data_directory.h"
#include "common/load_test_suites.h"

#include <oklt/core/error.h>
#include <oklt/core/target_backends.h>
#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>
#include <oklt/util/format.h>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <fstream>


//// partial specialization (full specialization works too)
//NLOHMANN_JSON_NAMESPACE_BEGIN
//template <typename T>
//struct adl_serializer<std::optional<T>> {
//    static void to_json(json& j, const std::optional<T>& opt) {
//        if (opt == std::nullopt) {
//            j = nullptr;
//        } else {
//            j = *opt; // this will call adl_serializer<T>::to_json which will
//                       // find the free function to_json in T's namespace!
//        }
//    }

//    static void from_json(const json& j, std::optional<T>& opt) {
//        if (j.is_null()) {
//            opt = std::nullopt;
//        } else {
//            opt = j.template get<T>(); // same as above, but with
//                                        // adl_serializer<T>::from_json
//        }
//    }
//};
//NLOHMANN_JSON_NAMESPACE_END


using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace oklt::tests;

enum struct Action { NORMALIZER, TRANSPILER, NORMALIZE_AND_TRANSPILE };
enum struct Compare { SOURCE, METADATA, ERROR_MESSAGE };

std::string toLower(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    std::transform(str.begin(), str.end(), std::back_inserter(result), ::tolower);
    return result;
}

std::string toCamelCase(std::string str) {
    std::size_t res_ind = 0;
    for (int i = 0; i < str.length(); i++) {
        // check for spaces in the sentence
        if (str[i] == ' ' || str[i] == '_') {
            // conversion into upper case
            str[i + 1] = ::toupper(str[i + 1]);
            continue;
        }
        // If not space, copy character
        else {
            str[res_ind++] = str[i];
        }
    }
    // return string to main
    return str.substr(0, res_ind);
}

tl::expected<Action, std::string> buildActionFrom(const std::string& v) {
    static const std::map<std::string, Action> actions = {
        {"normalizer", Action::NORMALIZER},
        {"transpiler", Action::TRANSPILER},
        {"normalize_and_transpile", Action::NORMALIZE_AND_TRANSPILE},
    };
    auto it = actions.find(toLower(v));
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
    auto it = compares.find(toLower(v));
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
    return oklt::UserInput{.backend = oklt::TargetBackend::CUDA, .source = std::move(sourceCode)};
}

struct TranspileActionConfig {
    std::string backend;
    std::filesystem::path source;
    std::vector<std::filesystem::path> mutable includes;
    std::vector<std::string> mutable defs;
    std::filesystem::path launcher;
    std::optional<std::string> intrinsic = std::nullopt;
    //NLOHMANN_DEFINE_TYPE_INTRUSIVE(TranspileActionConfig, backend, source, includes, defs, launcher, intrinsic)
    oklt::UserInput build(const fs::path& dataDir) const;
};

// void to_json(nlohmann::json& json, const TranspileActionConfig& dt);
void from_json(const json& j, TranspileActionConfig& conf) {

    j.at("backend").get_to(conf.backend);
    j.at("source").get_to(conf.source);
    j.at("includes").get_to(conf.includes);
    j.at("defs").get_to(conf.defs);
    j.at("launcher").get_to(conf.launcher);
    auto it = j.find("intrinsic");
    if(it != j.end()) {
        conf.intrinsic = it->get<std::string>();
    }
}

oklt::UserInput TranspileActionConfig::build(const fs::path& dataDir) const {
    auto expectedBackend = oklt::backendFromString(backend);
    if (!expectedBackend) {
        throw std::logic_error(expectedBackend.error());
    }
    auto sourceFullPath = dataDir / source;

    // add path to kernel source code headers lookup
    includes.emplace_back(sourceFullPath.parent_path().string());

    std::ifstream sourceFile{sourceFullPath};
    std::string sourceCode{std::istreambuf_iterator<char>(sourceFile), {}};

    std::vector<std::filesystem::path> intrinsics;
    if(intrinsic) {
        fs::path fullIntrinsicPath = dataDir / std::filesystem::path(intrinsic.value());
        intrinsics.push_back(std::move(fullIntrinsicPath));
    }
    return oklt::UserInput{.backend = expectedBackend.value(),
                           .source = std::move(sourceCode),
                           .sourcePath = std::move(sourceFullPath),
                           .includeDirectories = includes,
                           .defines = defs,
                           .userIntrinsics = std::move(intrinsics)
    };
}

namespace {
void compareError(const std::string& sourceFilePath,
                  const oklt::UserResult& res,
                  const std::string& refErrorMessage) {
    // TODO: currently compare only first error to the whole file. There can be
    // multiple errors
    EXPECT_EQ(res.has_value(), refErrorMessage.empty())
        << "No error when expected, or vice versa: " << sourceFilePath;
    ASSERT_FALSE(res.error().empty()) << "Failed silently? No error message: " << sourceFilePath;
    auto normalizeError = res.error().front();
    auto errorMessage = normalizeError.desc;

    // Since path to file in error message depends on pwd, we have to replace it with just filename
    // Error must start with sourceFilePath
    auto filename = fs::path(sourceFilePath).filename().string();
    if (errorMessage.find(sourceFilePath) == 0) {
        errorMessage.replace(0, sourceFilePath.size(), filename);
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
                SPDLOG_INFO("Run Normalize action for {}", input.sourcePath.string());
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
                            oklt::format(normalizeResult.value().normalized.source);
                        EXPECT_EQ(formatedReference, normalizedSource)
                            << fmt::format("Failed compare {} with generated normalized source",
                                           referencePath.string());
                        break;
                    }
                    case Compare::METADATA: {
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
                SPDLOG_INFO("Run Transpile action for {}", input.sourcePath.string());
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
                            oklt::format(transpileResult.value().kernel.source);
                        EXPECT_EQ(formatedReference, transpiledSource) << fmt::format(
                            "Failed compare {} with generated normalization metadataJson",
                            referencePath.string());

                        break;
                    }
                    case Compare::METADATA: {
                        EXPECT_EQ(reference, transpileResult.value().kernel.metadata)
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
                SPDLOG_INFO("Run Normalize and Transpile action for {}", input.sourcePath.string());
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
                            oklt::format(transpileResult.value().kernel.source);
                        EXPECT_EQ(formatedReference, transpiledSource) << fmt::format(
                            "Failed compare {} with generated normalized and transpiled",
                            referencePath.string());
                        break;
                    }
                    case Compare::METADATA: {
                        EXPECT_EQ(reference, transpileResult.value().kernel.metadata)
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
        return toCamelCase(fileName.string());
    }
};

INSTANTIATE_TEST_SUITE_P(GenericSuiteTests,
                         GenericTest,
                         testing::ValuesIn(loadTestsSuite()),
                         GenericConfigTestNamePrinter());
