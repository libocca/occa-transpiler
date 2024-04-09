#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

HandleResult AttrDeclHandler::handle(SessionStage& stage,
                                     const Decl& decl,
                                     const Attr& attr,
                                     const std::any* params) {
    return _handler(stage, decl, attr, params);
}

}  // namespace oklt
