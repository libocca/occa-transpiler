#pragma once

#include "okl_attr_marker.h"

#include <oklt/core/transpiler_session/transpiler_session.h>

#include <list>
#include <tl/expected.hpp>

using namespace clang;
namespace oklt {

struct OklToGnuStageOutput {
    std::string gnuCppSrc;
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
