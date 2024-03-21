#pragma once

#include "core/transpiler_session/transpiler_session.h"
#include "pipeline/stages/normalizer/impl/okl_attr_marker.h"

#include <tl/expected.hpp>

#include <list>

using namespace clang;
namespace oklt {

struct OklMacroStageOutput {
    std::string cppSrc;
    TransformedFiles cppIncs;
    SharedTranspilerSession session;
};

struct OklMacroStageInput {
    std::string cppSrc;
    SharedTranspilerSession session;
};

struct Error;
using OklMacroResult = tl::expected<OklMacroStageOutput, std::vector<Error>>;
OklMacroResult convertOklMacroAttribute(OklMacroStageInput input);
}  // namespace oklt
