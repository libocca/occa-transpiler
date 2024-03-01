#include <oklt/core/kernel_metadata.h>
#include "attributes/frontend/params/loop.h"

#include <string>

namespace oklt::dpcpp {
std::string dimToStr(const Dim& dim);
std::string getIdxVariable(const AttributedLoop& loop);
std::string buildInnerOuterLoopIdxLine(const LoopMetaData& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter);
}  // namespace oklt::dpcpp
