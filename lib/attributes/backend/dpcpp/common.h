#include "core/rewriter/rewriter_proxy.h"
#include "attributes/frontend/params/loop.h"

#include <string>


namespace clang {
class Rewriter;
}

namespace oklt {
struct OklLoopInfo;
}

namespace oklt::dpcpp {
std::string axisToStr(const Axis& axis);
std::string getIdxVariable(const AttributedLoop& loop);
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       oklt::Rewriter& rewriter);

const std::string SYNC_THREADS_BARRIER = "item_.barrier(sycl::access::fence_space::local_space)";
}  // namespace oklt::dpcpp
