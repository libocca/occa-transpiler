#pragma once

#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {

template <class TraverseType>
struct DefaultTraverseSema {
  bool beforeTraverse(TraverseType* node, SessionStage& stage) { return true; }

  bool afterTraverse(TraverseType* node, SessionStage& stage) { return true; }
};
}  // namespace oklt
