#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

HandleResult AttrHandler::handle(SessionStage& stage,
                                 const DynTypedNode& node,
                                 const Attr& attr,
                                 const std::any* params) {
    return _handler(stage, node, attr, params);
}

}  // namespace oklt
