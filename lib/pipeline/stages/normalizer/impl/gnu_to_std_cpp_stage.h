#pragma once

#include "core/transpiler_session/transpiler_session.h"
#include "pipeline/stages/normalizer/impl/okl_attr_marker.h"

#include <list>

#include <tl/expected.hpp>

namespace oklt {

struct GnuToStdCppStageOutput {
    std::string stdCppSrc;
    std::map<std::string, std::string> allStdCppSrcs;
    SharedTranspilerSession session;
};

struct GnuToStdCppStageInput {
    std::string gnuCppSrc;
    std::map<std::string, std::string> allGnuCppSrcs;
    std::list<OklAttrMarker> gnuMarkers;
    std::list<OklAttrMarker> recoveryMarkers;
    SharedTranspilerSession session;
};

struct Error;
using GnuToStdCppResult = tl::expected<GnuToStdCppStageOutput, std::vector<Error>>;
GnuToStdCppResult convertGnuToStdCppAttribute(GnuToStdCppStageInput input);
}  // namespace oklt
