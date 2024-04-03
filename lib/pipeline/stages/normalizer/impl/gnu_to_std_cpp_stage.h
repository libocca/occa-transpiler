#pragma once

#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/transpiler_session.h"
#include "pipeline/stages/normalizer/impl/okl_attr_marker.h"

#include <list>

#include <tl/expected.hpp>

namespace oklt {

struct GnuToStdCppStageOutput {
    std::string stdCppSrc;
    TransformedFiles stdCppIncs;
    SharedTranspilerSession session;
};

struct GnuToStdCppStageInput {
    std::string gnuCppSrc;
    TransformedFiles gnuCppIncs;
    std::list<OklAttrMarker> gnuMarkers;
    SharedTranspilerSession session;
};

struct Error;
using GnuToStdCppResult = tl::expected<GnuToStdCppStageOutput, std::vector<Error>>;
GnuToStdCppResult convertGnuToStdCppAttribute(GnuToStdCppStageInput input);
}  // namespace oklt
