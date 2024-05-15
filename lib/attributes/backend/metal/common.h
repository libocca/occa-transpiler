#include "attributes/attribute_names.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/default_handlers.h"
#include "attributes/utils/kernel_utils.h"
#include "attributes/utils/utils.h"
#include "core/handler_manager/backend_handler.h"
#include "core/rewriter/rewriter_proxy.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <string>

namespace clang {
class Rewriter;
}

namespace oklt {
struct OklLoopInfo;
}

namespace oklt::metal {
std::string axisToStr(const Axis& axis);
std::string getIdxVariable(const AttributedLoop& loop);
std::string getTiledVariableName(const OklLoopInfo& forLoop);

// Produces something like: int i = start +- (inc * _occa_group_position.x);
//                      or: int i = start +- (inc * _occa_thread_position.x);
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       oklt::Rewriter& rewriter);

const std::string SYNC_THREADS_BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup)";
}  // namespace oklt::metal
