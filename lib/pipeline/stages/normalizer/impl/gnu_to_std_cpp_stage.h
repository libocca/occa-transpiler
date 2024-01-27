#pragma once

#include <oklt/core/transpiler_session/transpiler_session.h>
#include "okl_attr_marker.h"

#include <list>
#include <tl/expected.hpp>

namespace oklt {

struct GnuToStdCppStageOutput {
  std::string stdCppSrc;
};

struct GnuToStdCppStageInput {
  std::string gnuCppSrc;
  std::list<OklAttrMarker> gnuMarkers;
  std::list<OklAttrMarker> recoveryMarkers;
};

tl::expected<GnuToStdCppStageOutput, int> convertGnuToStdCppAttribute(GnuToStdCppStageInput input,
                                                                      TranspilerSession& session);
}  // namespace oklt
