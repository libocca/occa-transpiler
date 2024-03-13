#pragma once

#include "core/transpiler_session/transpiler_session.h"
#include "pipeline/stages/normalizer/impl/okl_attr_marker.h"

#include <tl/expected.hpp>

#include <list>

using namespace clang;
namespace oklt {

struct OklToGnuStageOutput {
    std::string gnuCppSrc;
    TransformedFiles gnuCppIncs;
    std::list<OklAttrMarker> gnuMarkers;
    std::list<OklAttrMarker> recoveryMarkers;
    SharedTranspilerSession session;
};

struct OklToGnuStageInput {
    std::string oklCppSrc;
    SharedTranspilerSession session;
};

struct Error;
using OklToGnuResult = tl::expected<OklToGnuStageOutput, std::vector<Error>>;
OklToGnuResult convertOklToGnuAttribute(OklToGnuStageInput input);
}  // namespace oklt
