#include <oklt/core/error.h>

#include "pipeline/core/stage_action_registry.h"
#include "pipeline/core/stage_action_runner.h"
#include "pipeline/stages/normalizer/error_codes.h"

#include <clang/Tooling/Tooling.h>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace {
using namespace llvm;
using namespace clang;
using namespace clang::tooling;
}  // namespace

namespace oklt {

SharedTranspilerSessionResult runStageAction(StringRef stageName, SharedTranspilerSession session) {
    if (session->src.empty()) {
        SPDLOG_ERROR("Input source string is empty");
        auto error =
            makeError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "input source string is empty");
        return tl::make_unexpected(std::vector<Error>{error});
    }

    SPDLOG_INFO("start: {}", stageName);

    Twine toolName = stageName;

    auto& input = session->input;
    auto cppFileNamePath = input.sourcePath;
    auto cppFileName = std::string(cppFileNamePath.replace_extension(".cpp"));

    // TODO get this info from user input aka json prop file
    std::vector<std::string> args = {
        "-std=c++17", "-Wno-extra-tokens", "-Wno-invalid-pp-token", "-fparse-all-comments", "-I."};

    for (const auto& define : input.defines) {
        std::string def = "-D" + define;
        args.push_back(std::move(def));
    }

    for (const auto& includePath : input.inlcudeDirectories) {
        std::string incPath = "-I" + includePath.string();
        args.push_back(std::move(incPath));
    }

    auto stageAction = instantiateStageAction(stageName);
    if (!stageAction) {
        Error err{std::error_code(), fmt::format("no stage action: {} in registry", stageName)};
        return tl::make_unexpected(std::vector<Error>{err});
    }

    auto ok = stageAction->setSession(session);
    if (!ok) {
        Error err{std::error_code(),
                  fmt::format("failed to set input for stage action: {}", stageName)};
        return tl::make_unexpected(std::vector<Error>{err});
    }

    Twine code(session->src);
    bool ret = runToolOnCodeWithArgs(std::move(stageAction),
                                     code,
                                     args,
                                     cppFileName,
                                     toolName,
                                     std::make_shared<PCHContainerOperations>());

    // TODO make reporting of warnings as runtime option
    const auto& warnings = session->getWarnings();
    if (!warnings.empty()) {
        SPDLOG_INFO("{} warnings: ", stageName);
        for (const auto& w : warnings) {
            llvm::outs() << w.desc << "\n";
        }
    }
    if (!ret || !session->getErrors().empty()) {
        return tl::make_unexpected(std::move(session->getErrors()));
    }

    return session;
}

SharedTranspilerSessionResult runPipeline(const std::vector<std::string>& pipeline,
                                          SharedTranspilerSession session) {
    for (const auto& stage : pipeline) {
        auto result = runStageAction(stage, session);
        if (!result) {
            return tl::make_unexpected(result.error());
        }
        session = std::move(result.value());
    }

    return session;
}
}  // namespace oklt
