#pragma once

#include "okl_attr_marker.h"

#include <oklt/core/transpiler_session/transpiler_session.h>

#include <list>
#include <tl/expected.hpp>

namespace oklt {

struct GnuToStdCppStageOutput {
    std::string stdCppSrc;
    SharedTranspilerSession session;
};

struct GnuToStdCppStageInput {
    std::string gnuCppSrc;
    std::list<OklAttrMarker> gnuMarkers;
    std::list<OklAttrMarker> recoveryMarkers;
    SharedTranspilerSession session;
};

struct Error;
using GnuToStdCppResult = tl::expected<GnuToStdCppStageOutput, std::vector<Error>>;
GnuToStdCppResult convertGnuToStdCppAttribute(GnuToStdCppStageInput input);
}  // namespace oklt
