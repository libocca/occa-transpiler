#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool AttrDeclHandler::handle(const Attr* attr, const Decl* decl, SessionStage& stage) {
    return _handler(attr, decl, stage);
}

}  // namespace oklt
