#include <oklt/core/error.h>

#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/transpiler_session/session_result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"
#include "tl/expected.hpp"

#include <clang/Tooling/Tooling.h>

#include <llvm/Support/raw_os_ostream.h>
#include <oklt/util/format.h>

#include <spdlog/spdlog.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

TranspilerSessionResult runTranspilerStage(SharedTranspilerSession session) {
    SPDLOG_INFO("Start transpilation stage");
    auto& input = session->input;

    Twine tool_name = "okl-transpiler";
    // INFO: hot fix for *.okl extention
    auto cppFileNamePath = input.sourcePath;
    auto cppFileName = std::string(cppFileNamePath.replace_extension(".cpp"));

    // TODO get this info from user input aka json prop file
    std::vector<std::string> args = {"-std=c++17", "-Wno-extra-tokens", "-Wno-invalid-pp-token", "-fparse-all-comments", "-I."};

    for (const auto& define : input.defines) {
        std::string def = "-D" + define;
        args.push_back(std::move(def));
    }

    for (const auto& includePath : input.inlcudeDirectories) {
        std::string incPath = "-I" + includePath.string();
        args.push_back(std::move(incPath));
    }

    Twine code(input.sourceCode);

    bool ret = runToolOnCodeWithArgs(std::make_unique<oklt::TranspileFrontendAction>(*session),
                                     code,
                                     args,
                                     cppFileName,
                                     tool_name,
                                     std::make_shared<PCHContainerOperations>());

    // TODO make reporting of warnings as runtime option
    const auto& warnings = session->getWarnings();
    if (!warnings.empty()) {
        SPDLOG_INFO("Transpilation warnings: ");
        for (const auto& w : warnings) {
            llvm::outs() << w.desc << "\n";
        }
    }
    if (!ret || !session->getErrors().empty()) {
        return tl::make_unexpected(std::move(session->getErrors()));
    }

    session->output.kernel.sourceCode = oklt::format(session->output.kernel.sourceCode);

    if (!session->output.launcher.sourceCode.empty()) {
        session->output.launcher.sourceCode = oklt::format(session->output.launcher.sourceCode);
    }

    SPDLOG_DEBUG("stage 3 cpp source:\n\n{}", session->output.kernel.sourceCode);

    return session;
}
}  // namespace oklt
