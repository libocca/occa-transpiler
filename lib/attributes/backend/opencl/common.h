#include "core/rewriter/rewriter_proxy.h"
#include "attributes/frontend/params/loop.h"

#include <string>


namespace clang {
class Rewriter;
}

namespace oklt {
struct OklLoopInfo;
}

namespace oklt::opencl {
std::string axisToStr(const Axis& axis);
std::string getIdxVariable(const AttributedLoop& loop);
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       oklt::Rewriter& rewriter);

const std::string SYNC_THREADS_BARRIER = "barrier(CLK_LOCAL_MEM_FENCE)";
}
