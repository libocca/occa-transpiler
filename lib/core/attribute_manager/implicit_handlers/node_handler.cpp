#include "core/attribute_manager/implicit_handlers/node_handler.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

NodeHandler::NodeHandler(HandleType h)
    : _handler(std::move(h)) {}

HandleResult NodeHandler::operator()(SessionStage& stage, const DynTypedNode& node) {
    return _handler(stage, node);
}

}  // namespace oklt
