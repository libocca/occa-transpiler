#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

tl::expected<std::any, Error> AttrDeclHandler::handle(const Attr* attr,
                                                      const Decl* decl,
                                                      const std::any* params,
                                                      SessionStage& stage) {
    return _handler(attr, decl, params, stage);
}

}  // namespace oklt
