#pragma once

#include "core/transpiler_session/transpiler_session.h"

#include <tl/expected.hpp>

using namespace clang;
namespace oklt {

struct ExpandMacroStageOutput {
    std::string cppSrc;
    TransformedFiles cppIncs;
    SharedTranspilerSession session;
};

struct ExpandMacroStageInput {
    std::string cppSrc;
    SharedTranspilerSession session;
};

struct Error;
using ExpandMacroResult = tl::expected<ExpandMacroStageOutput, std::vector<Error>>;
ExpandMacroResult expandMacro(ExpandMacroStageInput input);
}  // namespace oklt
